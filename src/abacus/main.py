from __future__ import annotations

import functools
import logging as _logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue as MPQueue
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from abacus import __version__
from abacus.config import config
from abacus.consensus import ConsensusCall, create_consensus_calls, update_flanking_labels_based_on_consensus
from abacus.filtering import filter_read_calls
from abacus.graph import (
    FilteredRead,
    ReadCall,
    get_read_calls,
)
from abacus.group_summary import calculate_final_group_summaries
from abacus.haplotyping import run_haplotyping, summarize_parameter_estimates
from abacus.locus import load_loci_from_json
from abacus.logging import logger, set_log_file_handler
from abacus.parameter_estimation import HeterozygousParameters, HomozygousParameters
from abacus.preprocess import get_reads_in_locus
from abacus.str_vcf import write_vcf
from abacus.utils import Haplotype, Sex

ascii_art = r"""
╔═══════════════════════════════════════════════════════════════════╗
║    =|=                                                   =--|--=  ║
║   =-|-=                    --- ~•~ ---                    =-|-=   ║
║  =--|--=      _     ___     _      ___   _   _    ___      =|=    ║
║   =-|-=      / \   | _ )   / \    / __/ | | | |  / __/    =-|-=   ║
║    =|=      / _ \  | _ \  / _ \  | (__  | |_| |  \__ \   =--|--=  ║
║   =-|-=    /_/ \_\ |___/ /_/ \_\  \___\  \___/   \___/    =-|-=   ║
║  =--|--=                                                   =|=    ║
║   =-|-=                    --- ~•~ ---                    =-|-=   ║
║    =|=                                                   =--|--=  ║
╚═══════════════════════════════════════════════════════════════════╝
"""


# Define the help panels
INPUTS = "Inputs"
OUTPUTS = "Outputs"
QC_OPTIONS = "Quality Control"
OPTIONS = "Other Options"
CONFIGURATION = "Algorithm Configuration"


# Set up the CLI
app = typer.Typer(
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"Abacus version {__version__}")
        raise typer.Exit()


_locus_context: dict[str, str] = {"id": ""}


class _LocusFilter(_logging.Filter):
    """Prepends [locus_id] to log records that don't already contain the locus ID."""

    def filter(self, record: _logging.LogRecord) -> bool:
        locus_id = _locus_context["id"]
        if locus_id and locus_id not in str(record.msg):
            record.msg = f"[{locus_id}] {record.msg}"
        return True


def _worker_init(config_dict: dict, log_queue: MPQueue) -> None:
    """Initializer for worker processes: set up queue logging and restore config."""
    from abacus.config import config as _config

    root = _logging.getLogger()
    root.handlers.clear()
    root.addHandler(QueueHandler(log_queue))
    for k, v in config_dict.items():
        setattr(_config, k, v)


def _process_locus(locus, bam: Path, ref: Path, sex: Sex) -> dict:
    """Process a single locus and return all results as a dict."""
    _locus_context["id"] = locus.id
    locus_t0 = time.perf_counter()
    logger.info("Locus: %s  |  %s  |  %s:%d-%d", locus.id, locus.structure, locus.location.chrom, locus.location.start, locus.location.end)

    # Get reads in locus
    t0 = time.perf_counter()
    reads = get_reads_in_locus(bam, locus, ref)
    logger.debug(f"[TIMING] {locus.id} get_reads_in_locus: {time.perf_counter() - t0:.3f}s  ({len(reads)} reads)")

    # Handle ploidy
    if len(reads) < config.min_haplotyping_depth:
        logger.warning(f"Low coverage for locus {locus.id}. Setting ploidy to 1.")
        ploidy = 1
    elif locus.location.chrom == "chrY":
        ploidy = sex.value.count("Y")
    elif locus.location.chrom == "chrX":
        ploidy = sex.value.count("X")
    else:
        ploidy = 2

    # Call STR in individual reads
    t0 = time.perf_counter()
    read_calls, unmapped_reads = get_read_calls(reads, locus)
    logger.debug(f"[TIMING] {locus.id} get_read_calls: {time.perf_counter() - t0:.3f}s  ({len(read_calls)} calls, {len(unmapped_reads)} unmapped)")

    # Filter read calls
    t0 = time.perf_counter()
    good_read_calls, removed_read_calls = filter_read_calls(read_calls=read_calls)
    logger.debug(f"[TIMING] {locus.id} filter_read_calls: {time.perf_counter() - t0:.3f}s  ({len(good_read_calls)} kept, {len(removed_read_calls)} removed)")

    # Group read calls
    t0 = time.perf_counter()
    grouped_read_calls, outlier_read_calls, het_params, hom_params, test_summary_res_df = run_haplotyping(
        read_calls=good_read_calls,
        ploidy=ploidy,
    )
    logger.debug(
        f"[TIMING] {locus.id} run_haplotyping: {time.perf_counter() - t0:.3f}s  ({len(grouped_read_calls)} grouped, {len(outlier_read_calls)} outliers)",
    )

    removed_read_calls.extend(outlier_read_calls)
    locus_is_het = grouped_read_calls[0].haplotype in [Haplotype.H1, Haplotype.H2] if grouped_read_calls else False

    parameter_summary_df = summarize_parameter_estimates(het_params, hom_params)

    # Create raw consensus for each haplotype
    t0 = time.perf_counter()
    unique_haplotypes = {r.haplotype for r in grouped_read_calls}
    raw_consensus_calls: list[ConsensusCall] = []
    for haplotype in unique_haplotypes:
        haplotyped_read_calls = [r for r in grouped_read_calls if r.haplotype == haplotype]
        raw_consensus_calls.extend(create_consensus_calls(read_calls=haplotyped_read_calls, haplotype=haplotype))

    # Re-group flanking read calls based on the raw consensus
    grouped_read_calls = update_flanking_labels_based_on_consensus(
        read_calls=grouped_read_calls,
        consensus_read_calls=raw_consensus_calls,
    )

    # Create final consensus for each haplotype
    unique_haplotypes = {r.haplotype for r in grouped_read_calls}
    final_consensus_calls: list[ConsensusCall] = []
    for haplotype in unique_haplotypes:
        haplotyped_read_calls = [r for r in grouped_read_calls if r.haplotype == haplotype]
        final_consensus_calls.extend(create_consensus_calls(read_calls=haplotyped_read_calls, haplotype=haplotype))
    logger.debug(f"[TIMING] {locus.id} consensus: {time.perf_counter() - t0:.3f}s")

    grouped_read_calls.extend(removed_read_calls)
    haplotyping_df = calculate_final_group_summaries(grouped_read_calls)
    test_summary_res_df["locus_id"] = locus.id

    satellite_df_list = [
        pd.DataFrame(
            {"locus_id": locus.id, "idx": sat_idx, "satellite": "|".join(locus.satellites[sat_idx].sequences)},
            index=[0],
        )
        for sat_idx in range(len(locus.satellites))
    ]
    satellite_df = pd.concat(satellite_df_list)
    haplotyping_df = haplotyping_df.merge(satellite_df, on="idx", how="left")
    parameter_summary_df = parameter_summary_df.merge(satellite_df, on="idx", how="left")

    logger.debug(f"[TIMING] {locus.id} TOTAL: {time.perf_counter() - locus_t0:.3f}s")
    _locus_context["id"] = ""

    return {
        "locus_id": locus.id,
        "grouped_read_calls": grouped_read_calls,
        "unmapped_reads": unmapped_reads,
        "het_params": het_params,
        "hom_params": hom_params,
        "locus_is_het": locus_is_het,
        "final_consensus_calls": final_consensus_calls,
        "haplotyping_df": haplotyping_df,
        "test_summary_res_df": test_summary_res_df,
        "parameter_summary_df": parameter_summary_df,
    }


@app.command(
    help="[bold]Abacus[/bold]: A tool for STR genotyping, haplotyping and visualization 🧬",
    no_args_is_help=True,
)
def abacus(
    # Inputs
    bam: Annotated[
        Path,
        typer.Option(
            "--bam",
            "-i",
            help="Input BAM or CRAM file",
            rich_help_panel=INPUTS,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    ref: Annotated[
        Path,
        typer.Option(
            "--ref",
            "-r",
            help="Reference genome FASTA file",
            rich_help_panel=INPUTS,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    str_catalog: Annotated[
        Path,
        typer.Option(
            "--str-catalog",
            "-s",
            help="STR catalog JSON file",
            rich_help_panel=INPUTS,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    # Outputs
    report: Annotated[
        Path,
        typer.Option(
            "--report",
            "-o",
            help="Output HTML report",
            rich_help_panel=OUTPUTS,
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ],
    vcf: Annotated[
        Path,
        typer.Option(
            "--vcf",
            "-v",
            help="Output VCF file",
            rich_help_panel=OUTPUTS,
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ],
    # Options
    sample_id: Annotated[
        str,
        typer.Option(
            "--sample-id",
            "-n",
            help="Sample ID",
            rich_help_panel=OPTIONS,
        ),
    ],
    loci_subset: Annotated[
        list[str] | None,
        typer.Option(
            "--loci-subset",
            help="Subset of loci to process. If not provided, all loci will be processed. Use multiple times to specify multiple loci.",
            rich_help_panel=OPTIONS,
        ),
    ] = None,
    sex: Annotated[
        Sex,
        typer.Option(
            "--sex",
            help="Sex of the sample.",
            rich_help_panel=OPTIONS,
            case_sensitive=False,
        ),
    ] = Sex.XX,
    log_file: Annotated[
        Path,
        typer.Option(
            "--log-file",
            "-l",
            help="Log file",
            rich_help_panel=OPTIONS,
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ] = config.log_file,
    keep_temp_files: Annotated[
        bool,
        typer.Option(
            "--keep-temp-files",
            help="Keep temporary files",
            rich_help_panel=OPTIONS,
        ),
    ] = False,
    add_consensus_to_vcf: Annotated[
        bool,
        typer.Option(
            "--add-consensus-to-vcf",
            help="Add consensus calls to VCF file",
            rich_help_panel=OPTIONS,
        ),
    ] = config.add_consensus_to_vcf,
    add_contracted_consensus_to_vcf: Annotated[
        bool,
        typer.Option(
            "--add-contracted-consensus-to-vcf",
            help="Add contracted consensus calls to VCF file",
            rich_help_panel=OPTIONS,
        ),
    ] = config.add_contracted_consensus_to_vcf,
    # QC
    min_mean_str_quality: Annotated[
        int,
        typer.Option(
            "--min-str-qual",
            help="Minimum mean base quality in STR region",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.min_mean_str_quality,
    tol_mean_str_quality: Annotated[
        int,
        typer.Option(
            "--tol-str-qual",
            help="Tolerance for mean base quality in STR region",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.tol_mean_str_quality,
    min_q10_str_quality: Annotated[
        int,
        typer.Option(
            "--min-q10-str-quality",
            help="Minimum Q10 base quality in STR region",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.min_q10_str_quality,
    tol_q10_str_quality: Annotated[
        int,
        typer.Option(
            "--tol-q10-str-quality",
            help="Tolerance for Q10 base quality in STR region",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.tol_q10_str_quality,
    max_error_rate: Annotated[
        float,
        typer.Option(
            "--max-error-rate",
            help="Maximum allowed error rate in STR region",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.max_error_rate,
    tol_error_rate: Annotated[
        float,
        typer.Option(
            "--tol-error-rate",
            help="Tolerance for error rate in STR region",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.tol_error_rate,
    max_ref_divergence: Annotated[
        float,
        typer.Option(
            "--max-ref-divergence",
            help="Maximum allowed reference divergence in STR region",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.max_ref_divergence,
    min_n_outlier_detection: Annotated[
        int,
        typer.Option(
            "--min-n-outlier-detection",
            help="Minimum number of read calls to perform outlier detection",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.min_n_outlier_detection,
    # Configuration
    anchor_length: Annotated[
        int,
        typer.Option(
            "--anchor-length",
            help="Length of the left and right anchor sequences",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.anchor_len,
    min_end_qual: Annotated[
        int,
        typer.Option(
            "--min-end-qual",
            help="Minimum base quality at read ends. Used for trimming.",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.min_end_qual,
    min_anchor_overlap: Annotated[
        int,
        typer.Option(
            "--min-anchor-overlap",
            help="Minimum overlap between read and anchor",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.min_anchor_overlap,
    trim_window_size: Annotated[
        int,
        typer.Option(
            "--trim-window-size",
            help="Window size for trimming low quality bases",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.trim_window_size,
    max_trim: Annotated[
        int,
        typer.Option(
            "--max-trim",
            help="Maximum number of bases to trim from read ends",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.max_trim,
    min_haplotyping_depth: Annotated[
        int,
        typer.Option(
            "--min-haplotyping-depth",
            help="Minimum allowed depth for haplotyping. If depth is lower, locus is analyzed as homozygous.",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.min_haplotyping_depth,
    heterozygozity_alpha: Annotated[
        float,
        typer.Option(
            "--heterozygozity-alpha",
            help="Sensitivity cutoff for heterozygosity test. This test focuses on difference in length between haplotypes.",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.het_alpha,
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_flag=True,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Print debug timing info for each processing step",
            rich_help_panel=OPTIONS,
        ),
    ] = False,
    threads: Annotated[
        int,
        typer.Option(
            "--threads",
            "-t",
            help="Number of parallel worker processes (default: 1)",
            rich_help_panel=OPTIONS,
        ),
    ] = 1,
) -> None:

    # Setup logging to file
    set_log_file_handler(logger, log_file)

    # Add locus-context filter so parallel log lines can be identified/filtered
    logger.addFilter(_LocusFilter())

    # Enable DEBUG output on console when verbose flag is set
    if verbose:
        for handler in logger.handlers:
            if isinstance(handler, _logging.StreamHandler) and not isinstance(handler, _logging.FileHandler):
                handler.setLevel(_logging.DEBUG)

    # Setup configuration
    config.anchor_len = anchor_length

    config.min_anchor_overlap = min_anchor_overlap
    config.min_end_qual = min_end_qual
    config.trim_window_size = trim_window_size
    config.max_trim = max_trim
    config.min_haplotyping_depth = min_haplotyping_depth
    config.het_alpha = heterozygozity_alpha

    # QC
    config.min_mean_str_quality = min_mean_str_quality
    config.tol_mean_str_quality = tol_mean_str_quality
    config.min_q10_str_quality = min_q10_str_quality
    config.tol_q10_str_quality = tol_q10_str_quality
    config.max_error_rate = max_error_rate
    config.tol_error_rate = tol_error_rate
    config.max_ref_divergence = max_ref_divergence

    config.min_n_outlier_detection = min_n_outlier_detection

    # VCF options
    config.add_consensus_to_vcf = add_consensus_to_vcf
    config.add_contracted_consensus_to_vcf = add_contracted_consensus_to_vcf

    # Welcome message
    logger.info(ascii_art)

    # Load loci data from JSON
    loci = load_loci_from_json(str_catalog, ref)

    # Subset loci if provided
    if loci_subset:
        if set(loci_subset).isdisjoint([locus.id for locus in loci]):
            logger.error("No loci in subset found in STR catalog")
            raise typer.Exit(code=1)

        loci = [locus for locus in loci if locus.id in loci_subset]

    # Process each locus (in parallel if --threads > 1)
    logger.info("Processing loci...")
    process_fn = functools.partial(_process_locus, bam=bam, ref=ref, sex=sex)

    if threads == 1:
        results = [process_fn(locus) for locus in loci]
    else:
        log_queue: MPQueue = MPQueue()
        queue_listener = QueueListener(log_queue, *logger.handlers, respect_handler_level=True)
        queue_listener.start()
        try:
            with ProcessPoolExecutor(
                max_workers=threads,
                initializer=_worker_init,
                initargs=(config.to_dict(), log_queue),
            ) as executor:
                results = list(executor.map(process_fn, loci))
        finally:
            queue_listener.stop()

    # Aggregate results from all loci (order preserved by executor.map)
    het_params_dict: dict[str, HeterozygousParameters] = {}
    hom_params_dict: dict[str, HomozygousParameters] = {}
    locus_is_het_dict: dict[str, bool] = {}
    all_read_calls: list[ReadCall] = []
    all_filtered_reads: list[FilteredRead] = []
    all_consensus_calls: list[ConsensusCall] = []
    all_haplotyping_df: list[pd.DataFrame] = []
    all_summaries_df: list[pd.DataFrame] = []
    all_par_summaries_df: list[pd.DataFrame] = []

    for result in results:
        locus_id = result["locus_id"]
        het_params_dict[locus_id] = result["het_params"]
        hom_params_dict[locus_id] = result["hom_params"]
        locus_is_het_dict[locus_id] = result["locus_is_het"]
        all_read_calls.extend(result["grouped_read_calls"])
        all_filtered_reads.extend(result["unmapped_reads"])
        all_consensus_calls.extend(result["final_consensus_calls"])
        all_haplotyping_df.append(result["haplotyping_df"])
        all_summaries_df.append(result["test_summary_res_df"])
        all_par_summaries_df.append(result["parameter_summary_df"])

    # Create output directory
    tmp_dir = report.parent / f"tmp_abacus_{sample_id}"
    tmp_dir.mkdir(exist_ok=True)

    # Write output files
    reads_csv = tmp_dir / "reads.csv"
    filtered_reads_csv = tmp_dir / "filtered_reads.csv"
    consensus_csv = tmp_dir / "consensus.csv"

    haplotypes_csv = tmp_dir / "haplotypes.csv"
    summary_csv = tmp_dir / "summary.csv"
    par_summary_csv = tmp_dir / "par_summary.csv"

    with Path.open(reads_csv, "w") as f:
        pd.DataFrame([r.to_dict() for r in all_read_calls]).to_csv(f, index=False)
    with Path.open(filtered_reads_csv, "w") as f:
        pd.DataFrame([r.to_dict() for r in all_filtered_reads]).to_csv(f, index=False)
    with Path.open(consensus_csv, "w") as f:
        pd.DataFrame([c.to_dict() for c in all_consensus_calls]).to_csv(f, index=False)

    with Path.open(haplotypes_csv, "w") as f:
        pd.concat(all_haplotyping_df).to_csv(f, index=False)
    with Path.open(summary_csv, "w") as f:
        pd.concat(all_summaries_df).to_csv(f, index=False)
    with Path.open(par_summary_csv, "w") as f:
        pd.concat(all_par_summaries_df).to_csv(f, index=False)

    # Write VCF output
    write_vcf(
        vcf=vcf,
        consensus_calls=all_consensus_calls,
        reference=ref,
        sample_id=sample_id,
        het_params_dict=het_params_dict,
        hom_params_dict=hom_params_dict,
        locus_is_het_dict=locus_is_het_dict,
    )

    # Render report
    logger.info("Rendering report...")
    report_template = Path(__file__).parent / "scripts" / "report_template.Rmd"

    process = subprocess.run(
        [
            "Rscript",
            "-e",
            f"""
                    rmarkdown::render('{report_template}', \
                        output_file='{report.name}', \
                        output_dir='{report.parent}', \
                        intermediates_dir='{tmp_dir}', \
                        params=list( \
                            sample_id = '{sample_id}', \
                            input_bam = '{bam}', \
                            str_catalog = '{str_catalog}', \
                            reads_csv = '{reads_csv}', \
                            filtered_reads_csv = '{filtered_reads_csv}', \
                            consensus_csv = '{consensus_csv}', \
                            clustering_summary_csv = '{haplotypes_csv}', \
                            test_summary_csv = '{summary_csv}', \
                            par_summary_csv = '{par_summary_csv}', \
                            min_mean_str_quality = {config.min_mean_str_quality}, \
                            min_q10_str_quality = {config.min_q10_str_quality}, \
                            max_error_rate = {config.max_error_rate}, \
                            max_ref_divergence = {config.max_ref_divergence} \
                        ) \
                    ) \
                    """,
        ],
        text=True,
        check=False,
        capture_output=True,
    )

    if process.returncode != 0:
        logger.debug("Rscript stdout:\n%s", process.stdout)
        logger.debug("Rscript stderr:\n%s", process.stderr)
        logger.error("Rscript failed with error code %d", process.returncode)
        raise typer.Exit(code=1)

    if not keep_temp_files:
        logger.info("Cleaning up temporary files...")
        for file in tmp_dir.iterdir():
            file.unlink(missing_ok=True)
        tmp_dir.rmdir()

    logger.info("Finished!")


if __name__ == "__main__":
    app()
