from __future__ import annotations

import functools
import logging as _logging
import math
import subprocess
import time
from importlib.resources import files
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Pool
from multiprocessing import Queue as MPQueue
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from abacus import __version__
from abacus.config import config
from abacus.consensus import ConsensusCall, create_consensus_calls, update_flanking_labels_based_on_consensus
from abacus.filtering import filter_low_qual_read_calls, filter_outlier_qual_read_calls
from abacus.graph import (
    ReadCall,
    get_read_calls,
)
from abacus.group_summary import calculate_final_group_summaries
from abacus.haplotyping import run_haplotyping, summarize_final_parameter_estimates, summarize_test_parameter_estimates
from abacus.locus import load_loci_from_json
from abacus.logging import logger, set_log_file_handler
from abacus.preprocess import get_reads_in_locus
from abacus.str_vcf import create_vcf_records, write_vcf
from abacus.utils import Haplotype, Sex

ascii_art = r"""
╔═════════════════════════════════════════════════════════════════════════╗
║   ■─┼─□                                                         □─┼─■   ║
║  □──┼──■  ────@@@────@@──────────────────────────────────────    ○┼●    ║
║   ○─┼─●   ──@@───@@──@@@@@@───@@@@@───@@@@@──@@────@@──@@@@──   ●─┼─○   ║
║    □┼■    ─@@─────@@─@@───@@─@@───@@─@@───@@─@@────@@─@@───@─  □──┼──■  ║
║   □─┼─■   ─@@@@@@@@@─@@───@@─@@───@@─@@──────@@────@@───@@───   ■─┼─□   ║
║  ●──┼──○  ─@@─────@@─@@───@@─@@───@@─@@───@@─@@@───@@─@───@@─    ■┼□    ║
║   ■─┼─□   ─@@─────@@─@@@@@@───@@@@─@──@@@@@──@@─@@@@───@@@@──   ○─┼─●   ║
║    ●┼○  ─────────────────────────────────────────────────────  ●──┼──○  ║
║   ○─┼─●                                                         ■─┼─□   ║
╚═════════════════════════════════════════════════════════════════════════╝
"""


# Define the help panels
INPUTS = "Inputs"
OUTPUTS = "Outputs"
QC_OPTIONS = "Quality Control"
OPTIONS = "Other Options"
CONFIGURATION = "Algorithm Configuration"


def _default_catalog_path() -> Path:
    return Path(str(files("abacus").joinpath("str_catalogs/abacus_catalog.json")))


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


def show_catalog_callback(value: bool) -> None:
    if value:
        typer.echo(_default_catalog_path().read_text(), nl=False)
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

    # Initialize list to keep track of removed read calls for final summary
    all_removed_read_calls: list[ReadCall] = []

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

    # Prefilter low quality read calls
    t0 = time.perf_counter()
    good_read_calls, low_quality_read_calls = filter_low_qual_read_calls(read_calls=read_calls)
    all_removed_read_calls.extend(low_quality_read_calls)
    logger.debug(
        f"[TIMING] {locus.id} filter_low_qual_read_calls: {time.perf_counter() - t0:.3f}s  ({len(good_read_calls)} kept, {len(low_quality_read_calls)} removed)",
    )

    # First round haplotyping (includes singleton detection + length outlier detection + re-estimation internally)
    t0 = time.perf_counter()
    initial_grouped_read_calls, initial_haplotyping_outliers, _, _, _, _ = run_haplotyping(
        read_calls=good_read_calls,
        ploidy=ploidy,
    )
    all_removed_read_calls.extend(initial_haplotyping_outliers)
    logger.debug(
        f"[TIMING] {locus.id} run_haplotyping: {time.perf_counter() - t0:.3f}s  ({len(initial_grouped_read_calls)} grouped, {len(initial_haplotyping_outliers)} removed)",
    )

    # Filter QC outliers per haplotype group
    good_read_calls, outlier_quality_read_calls = filter_outlier_qual_read_calls(read_calls=initial_grouped_read_calls)
    all_removed_read_calls.extend(outlier_quality_read_calls)

    # Second round haplotyping (includes singleton detection + length outlier detection + re-estimation internally)
    t0 = time.perf_counter()
    grouped_read_calls, haplotyping_outliers, het_params, hom_params, final_params, test_summary_res_df = run_haplotyping(
        read_calls=good_read_calls,
        ploidy=ploidy,
    )
    all_removed_read_calls.extend(haplotyping_outliers)
    logger.debug(
        f"[TIMING] {locus.id} run_haplotyping: {time.perf_counter() - t0:.3f}s  ({len(grouped_read_calls)} grouped, {len(haplotyping_outliers)} removed)",
    )

    # TODO: Make this nicer
    locus_is_het = grouped_read_calls[0].haplotype in [Haplotype.H1, Haplotype.H2] if grouped_read_calls else False

    final_parameter_summary_df = summarize_final_parameter_estimates(final_params)
    test_parameter_summary_df = summarize_test_parameter_estimates(het_params, hom_params)

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

    grouped_read_calls.extend(all_removed_read_calls)
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
    final_parameter_summary_df = final_parameter_summary_df.merge(satellite_df, on="idx", how="left")
    test_parameter_summary_df = test_parameter_summary_df.merge(satellite_df, on="idx", how="left")

    # Generate VCF records here while final_consensus_calls, final_params, and locus_is_het are in scope,
    # so the main process never needs to accumulate the full ConsensusCall/params objects.
    vcf_records = create_vcf_records(final_consensus_calls, ref, final_params, locus_is_het)
    vcf_unique_alts = {int(v) for params in final_params.values() for v in params.mean if not math.isnan(v)}

    logger.debug(f"[TIMING] {locus.id} TOTAL: {time.perf_counter() - locus_t0:.3f}s")
    _locus_context["id"] = ""

    return {
        "locus_id": locus.id,
        "grouped_read_calls": grouped_read_calls,
        "unmapped_reads": unmapped_reads,
        "final_consensus_calls": final_consensus_calls,
        "haplotyping_df": haplotyping_df,
        "test_summary_res_df": test_summary_res_df,
        "final_parameter_summary_df": final_parameter_summary_df,
        "test_parameter_summary_df": test_parameter_summary_df,
        "vcf_records": vcf_records,
        "vcf_unique_alts": vcf_unique_alts,
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
    # Outputs
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
    report: Annotated[
        Path | None,
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
    ] = None,
    str_catalog: Annotated[
        Path | None,
        typer.Option(
            "--str-catalog",
            "-s",
            help="STR catalog JSON file [default: built-in abacus catalog]",
            rich_help_panel=INPUTS,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    loci_subset: Annotated[
        list[str] | None,
        typer.Option(
            "--loci-subset",
            help="Loci to process (repeatable). Defaults to all loci in the catalog.",
            rich_help_panel=OPTIONS,
        ),
    ] = None,
    loci_subset_file: Annotated[
        Path | None,
        typer.Option(
            "--loci-subset-file",
            help="Path to a file containing one locus ID per line to process.",
            rich_help_panel=OPTIONS,
            exists=True,
            file_okay=True,
            dir_okay=False,
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
            help="Keep temporary files after report generation",
            rich_help_panel=OPTIONS,
        ),
    ] = False,
    add_consensus_to_vcf: Annotated[
        bool,
        typer.Option(
            "--add-consensus-to-vcf",
            help="Include consensus sequence in VCF output",
            rich_help_panel=OPTIONS,
        ),
    ] = config.add_consensus_to_vcf,
    add_contracted_consensus_to_vcf: Annotated[
        bool,
        typer.Option(
            "--add-contracted-consensus-to-vcf",
            help="Include contracted consensus sequence in VCF output",
            rich_help_panel=OPTIONS,
        ),
    ] = config.add_contracted_consensus_to_vcf,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Enable verbose/debug logging",
            rich_help_panel=OPTIONS,
        ),
    ] = False,
    threads: Annotated[
        int,
        typer.Option(
            "--threads",
            "-t",
            min=1,
            help="Number of parallel worker processes",
            rich_help_panel=OPTIONS,
        ),
    ] = 1,
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
    min_n_qc_filtering: Annotated[
        int,
        typer.Option(
            "--min-n-qc-filtering",
            help="Minimum number of read calls required to run the statistical QC filtering step",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.min_n_qc_filtering,
    min_n_length_outlier_detection: Annotated[
        int,
        typer.Option(
            "--min-n-length-outlier-detection",
            help="Minimum number of spanning reads per haplotype group to perform length outlier detection",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.min_n_length_outlier_detection,
    tol_length_outlier_pct: Annotated[
        float,
        typer.Option(
            "--length-outlier-tolerance",
            help="Reads within this % of the haplotype median length are always retained during outlier removal",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.tol_length_outlier_pct,
    tol_length_outlier_bases: Annotated[
        int,
        typer.Option(
            "--length-outlier-tolerance-bases",
            help="Reads within this number of bases of the haplotype median length are always retained during outlier removal",
            rich_help_panel=QC_OPTIONS,
        ),
    ] = config.tol_length_outlier_bases,
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
    heterozygosity_alpha: Annotated[
        float,
        typer.Option(
            "--heterozygosity-alpha",
            help="Significance threshold for the length-based heterozygosity test",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.het_alpha,
    equal_length_alpha: Annotated[
        float,
        typer.Option(
            "--equal-length-alpha",
            help="Significance threshold for the sequence-based heterozygosity test (used when haplotypes have equal length)",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.equal_length_alpha,
    downsample: Annotated[
        int,
        typer.Option(
            "--downsample",
            help="Randomly downsample to this many reads per locus when coverage exceeds the threshold. Set to 0 to disable.",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.downsample,
    downsample_seed: Annotated[
        int,
        typer.Option(
            "--downsample-seed",
            help="Random seed for downsampling reproducibility.",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.downsample_seed,
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_flag=True,
        is_eager=True,
        help="Show version and exit.",
    ),
    _show_catalog: bool | None = typer.Option(
        None,
        "--show-catalog",
        callback=show_catalog_callback,
        is_flag=True,
        is_eager=True,
        help="Print the built-in STR catalog to stdout and exit.",
    ),
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
    config.het_alpha = heterozygosity_alpha
    config.equal_length_alpha = equal_length_alpha

    # QC
    config.min_mean_str_quality = min_mean_str_quality
    config.tol_mean_str_quality = tol_mean_str_quality
    config.min_q10_str_quality = min_q10_str_quality
    config.tol_q10_str_quality = tol_q10_str_quality
    config.max_error_rate = max_error_rate
    config.tol_error_rate = tol_error_rate
    config.max_ref_divergence = max_ref_divergence

    config.min_n_qc_filtering = min_n_qc_filtering
    config.min_n_length_outlier_detection = min_n_length_outlier_detection
    config.tol_length_outlier_pct = tol_length_outlier_pct
    config.tol_length_outlier_bases = tol_length_outlier_bases

    config.downsample = downsample
    config.downsample_seed = downsample_seed

    # VCF options
    config.add_consensus_to_vcf = add_consensus_to_vcf
    config.add_contracted_consensus_to_vcf = add_contracted_consensus_to_vcf

    # Welcome message
    logger.info(ascii_art)

    # Load loci data from JSON
    if str_catalog is None:
        str_catalog = _default_catalog_path()
    loci = load_loci_from_json(str_catalog, ref)

    # Subset loci if provided
    if loci_subset or loci_subset_file:
        # Combine locus IDs from command line and file, ensuring no duplicates
        selected_loci = list(loci_subset or [])
        if loci_subset_file:
            file_loci = {line.strip() for line in loci_subset_file.read_text().splitlines() if line.strip()}
            selected_loci.extend(file_loci)

        # Check that all loci are in the catalog
        loci_not_found = set(selected_loci) - {locus.id for locus in loci}
        if loci_not_found:
            logger.warning("Some loci in subset not found in STR catalog: %s", ", ".join(loci_not_found))
            raise typer.Exit(code=1)

        # Filter loci
        loci = [locus for locus in loci if locus.id in selected_loci]

    # Create tmp directory before processing so partial results are written on failure
    tmp_dir = vcf.parent / f"tmp_abacus_{sample_id}"
    tmp_dir.mkdir(exist_ok=True)

    reads_csv = tmp_dir / "reads.csv"
    filtered_reads_csv = tmp_dir / "filtered_reads.csv"
    consensus_csv = tmp_dir / "consensus.csv"
    haplotypes_csv = tmp_dir / "haplotypes.csv"
    summary_csv = tmp_dir / "summary.csv"
    final_param_summary_csv = tmp_dir / "final_parameter_summary.csv"
    test_params_summary_csv = tmp_dir / "test_parameter_summary.csv"
    vcf_records_tmp = tmp_dir / "vcf_records.tmp"

    # Remove any existing temp files from previous runs with the same sample ID to avoid appending to old results
    for _f in [reads_csv, filtered_reads_csv, consensus_csv, haplotypes_csv, summary_csv, final_param_summary_csv, test_params_summary_csv, vcf_records_tmp]:
        _f.unlink(missing_ok=True)

    def _append_to_csv(df: pd.DataFrame, path: Path) -> None:
        if df.empty:
            return
        df.to_csv(path, mode="a", header=not path.exists(), index=False)

    unique_alts: set[int] = set()

    def _handle_result(result: dict) -> None:
        _append_to_csv(pd.DataFrame([r.to_dict() for r in result["grouped_read_calls"]]), reads_csv)
        _append_to_csv(pd.DataFrame([r.to_dict() for r in result["unmapped_reads"]]), filtered_reads_csv)
        _append_to_csv(pd.DataFrame([c.to_dict() for c in result["final_consensus_calls"]]), consensus_csv)
        _append_to_csv(result["haplotyping_df"], haplotypes_csv)
        _append_to_csv(result["test_summary_res_df"], summary_csv)
        _append_to_csv(result["final_parameter_summary_df"], final_param_summary_csv)
        _append_to_csv(result["test_parameter_summary_df"], test_params_summary_csv)
        with vcf_records_tmp.open("a") as f:
            for line in result["vcf_records"]:
                if line:
                    f.write(line + "\n")
        unique_alts.update(result["vcf_unique_alts"])

    # Process each locus (in parallel if --threads > 1), writing results incrementally
    logger.info("Processing loci...")
    process_fn = functools.partial(_process_locus, bam=bam, ref=ref, sex=sex)

    if threads == 1:
        for locus in loci:
            _handle_result(process_fn(locus))
    else:
        log_queue: MPQueue = MPQueue()
        queue_listener = QueueListener(log_queue, *logger.handlers, respect_handler_level=True)
        queue_listener.start()
        try:
            with Pool(
                processes=threads,
                initializer=_worker_init,
                initargs=(config.to_dict(), log_queue),
                maxtasksperchild=100,  # recycle workers to release fragmented heap
            ) as pool:
                for result in pool.imap_unordered(process_fn, loci, chunksize=1):
                    _handle_result(result)
        finally:
            queue_listener.stop()

    # Write VCF output
    write_vcf(
        vcf=vcf,
        vcf_records_tmp=vcf_records_tmp,
        reference=ref,
        sample_id=sample_id,
        unique_alts=unique_alts,
    )

    # Render report (only if --report was provided)
    if report is not None:
        logger.info("Rendering report...")
        report_template = Path(__file__).parent / "scripts" / "report_template.Rmd"
        logo_path = Path(__file__).parent.parent.parent / "img" / "logo.png"

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
                                abacus_version = '{__version__}', \
                                sample_id = '{sample_id}', \
                                input_bam = '{bam}', \
                                str_catalog = '{str_catalog}', \
                                reads_csv = '{reads_csv}', \
                                filtered_reads_csv = '{filtered_reads_csv}', \
                                consensus_csv = '{consensus_csv}', \
                                clustering_summary_csv = '{haplotypes_csv}', \
                                test_summary_csv = '{summary_csv}', \
                                final_param_summary_csv = '{final_param_summary_csv}', \
                                test_param_summary_csv = '{test_params_summary_csv}', \
                                min_mean_str_quality = {config.min_mean_str_quality}, \
                                min_q10_str_quality = {config.min_q10_str_quality}, \
                                max_error_rate = {config.max_error_rate}, \
                                max_ref_divergence = {config.max_ref_divergence}, \
                                logo_path = '{logo_path}' \
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
