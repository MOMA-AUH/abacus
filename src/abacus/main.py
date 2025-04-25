from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from abacus.config import config
from abacus.consensus import ConsensusCall, create_consensus_calls, get_heterozygote_labels_seq
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
from abacus.preprocess import get_reads_in_locus
from abacus.str_vcf import write_vcf
from abacus.utils import Sex

ascii_art = r"""
╔═══════════════════════════════════════════════════════════════════╗
║    =|=                                                   =--|--=  ║
║   =-|-=                    --- ~•~ ---                    =-|-=   ║
║  =--|--=      _     ___     _      ___   _   _    ___      =|=    ║
║   =-|-=      / \   | _ )   / \    / __/ | | | |  / __|    =-|-=   ║
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
OPTIONS = "Other Options"
CONFIGURATION = "Algorithm Configuration"


# Set up the CLI
app = typer.Typer(
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


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
            help="Input BAM file",
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
    str_catalouge: Annotated[
        Path,
        typer.Option(
            "--str-catalouge",
            "-s",
            help="STR catalouge JSON file",
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
    # Configuration
    anchor_length: Annotated[
        int,
        typer.Option(
            "--anchor-length",
            help="Length of the left and right anchor sequences",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.anchor_len,
    min_anchor_overlap: Annotated[
        int,
        typer.Option(
            "--min-anchor-overlap",
            help="Minimum overlap between read and anchor",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.min_anchor_overlap,
    min_str_qual: Annotated[
        int,
        typer.Option(
            "--min-str-qual",
            help="Minimum median base quality in STR region",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.min_str_qual,
    min_end_qual: Annotated[
        int,
        typer.Option(
            "--min-end-qual",
            help="Minimum base quality at read ends. Used for trimming.",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.min_end_qual,
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
    error_rate_threshold: Annotated[
        float,
        typer.Option(
            "--error-rate-threshold",
            help="Threshold for error rate",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.error_rate_threshold,
    min_haplotype_depth: Annotated[
        int,
        typer.Option(
            "--min-haplotype-depth",
            help="Minimum allowed depth for each called haplotype. Else loci will be called as homozygous.",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.min_haplotype_depth,
    heterozygozity_alpha: Annotated[
        float,
        typer.Option(
            "--heterozygozity-alpha",
            help="Sensitivity cutoff for heterozygosity test. This test focuses on difference in length between haplotypes.",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.het_alpha,
    split_alpha: Annotated[
        float,
        typer.Option(
            "--split-alpha",
            help="Sensitivity cutoff for split haplotype test. This test focuses on difference in read depth between haplotypes.",
            rich_help_panel=CONFIGURATION,
        ),
    ] = config.split_alpha,
) -> None:
    # Setup logging to file
    set_log_file_handler(logger, log_file)

    # Setup configuration
    config.anchor_len = anchor_length
    config.min_anchor_overlap = min_anchor_overlap
    config.min_str_qual = min_str_qual
    config.min_end_qual = min_end_qual
    config.trim_window_size = trim_window_size
    config.max_trim = max_trim
    config.error_rate_threshold = error_rate_threshold
    config.min_haplotype_depth = min_haplotype_depth
    config.het_alpha = heterozygozity_alpha
    config.split_alpha = split_alpha

    # Welcome message
    # TODO: Update logs with welcome, and loci messages
    logger.info(ascii_art)

    # Load loci data from JSON
    loci = load_loci_from_json(str_catalouge, ref)

    # Check if loci subset is valid
    if loci_subset and set(loci_subset).isdisjoint([locus.id for locus in loci]):
        logger.error("No loci in subset found in STR catalouge")
        raise typer.Exit(code=1)

    # Initialize output data
    all_read_calls: list[ReadCall] = []
    all_filtered_reads: list[FilteredRead] = []
    all_consensus_calls: list[ConsensusCall] = []

    all_haplotyping_df: list[pd.DataFrame] = []
    all_summaries_df: list[pd.DataFrame] = []
    all_par_summaries_df: list[pd.DataFrame] = []

    # Process each locus
    logger.info("Processing loci...")
    for locus in loci:
        # Skip loci not in subset
        if loci_subset and locus.id not in loci_subset:
            continue

        logger.info("Current locus:")
        logger.info(f"- ID: {locus.id}")
        logger.info(f"- Structure: {locus.structure}")
        logger.info(f"- Position: {locus.location.chrom}:{locus.location.start}-{locus.location.end}")

        if not locus.satellites:
            logger.error("No valid satellite pattern found in STR definition")
            continue

        # Get reads in locus
        reads = get_reads_in_locus(bam, locus)

        # Get ploidy
        # Handle sex chromosomes
        if locus.location.chrom == "chrY":
            ploidy = sex.value.count("Y")
        elif locus.location.chrom == "chrX":
            ploidy = sex.value.count("X")
        else:
            ploidy = 2

        # If low coverage, set ploidy to 1
        if len(reads) < config.min_haplotype_depth * 2:
            ploidy = 1

        # Call STR in individual reads
        read_calls, unmapped_reads = get_read_calls(reads, locus)

        # Filter read calls
        # TODO: Go through filters and clean up unused and unnecessary filters
        good_read_calls, filtered_read_calls = filter_read_calls(read_calls=read_calls)

        # Group read calls
        grouped_read_calls, outlier_read_calls, het_params, hom_params, test_summary_res_df = run_haplotyping(
            read_calls=good_read_calls,
            ploidy=ploidy,
        )

        filtered_read_calls.extend(outlier_read_calls)

        # Summarize haplotype estimation
        parameter_summary_df = summarize_parameter_estimates(grouped_read_calls, het_params, hom_params)

        # Create raw consensus for each haplotype
        unique_haplotypes = {r.haplotype for r in grouped_read_calls}
        raw_consensus_calls: list[ConsensusCall] = []
        for haplotype in unique_haplotypes:
            haplotyped_read_calls = [r for r in grouped_read_calls if r.haplotype == haplotype]
            raw_consensus_calls.extend(create_consensus_calls(read_calls=haplotyped_read_calls, haplotype=haplotype))

        # Re-group flanking read calls based on the raw consensus
        grouped_read_calls = get_heterozygote_labels_seq(
            read_calls=grouped_read_calls,
            consensus_read_calls=raw_consensus_calls,
        )

        # Crerate final consensus for each haplotype
        unique_haplotypes = {r.haplotype for r in grouped_read_calls}
        final_consensus_calls: list[ConsensusCall] = []
        for haplotype in unique_haplotypes:
            haplotyped_read_calls = [r for r in grouped_read_calls if r.haplotype == haplotype]
            final_consensus_calls.extend(create_consensus_calls(read_calls=haplotyped_read_calls, haplotype=haplotype))

        # Add consensus calls to output
        all_consensus_calls.extend(final_consensus_calls)

        # Combine results
        grouped_read_calls.extend(filtered_read_calls)
        # Calculate final group summaries

        haplotyping_df = calculate_final_group_summaries(grouped_read_calls)

        # TODO: Remove this
        # Annotate results with locus info
        test_summary_res_df["locus_id"] = locus.id

        # Annotate results with Locus ID and Satellite sequence
        satellite_df_list = [
            pd.DataFrame(
                {
                    "locus_id": locus.id,
                    "idx": sat_idx,
                    "satellite": locus.satellites[sat_idx].sequence,
                },
                index=[0],
            )
            for sat_idx in range(len(locus.satellites))
        ]
        satellite_df = pd.concat(satellite_df_list)

        haplotyping_df = haplotyping_df.merge(satellite_df, on="idx", how="left")
        parameter_summary_df = parameter_summary_df.merge(satellite_df, on="idx", how="left")

        # Concatenate results
        all_read_calls.extend(grouped_read_calls)
        all_filtered_reads.extend(unmapped_reads)

        all_haplotyping_df.append(haplotyping_df)
        all_summaries_df.append(test_summary_res_df)
        all_par_summaries_df.append(parameter_summary_df)

    # Create output directory
    working_dir = report.parent / "abacus_output"
    working_dir.mkdir(exist_ok=True)

    # Write output files
    reads_csv = working_dir / "reads.csv"
    filtered_reads_csv = working_dir / "filtered_reads.csv"
    consensus_csv = working_dir / "consensus.csv"

    haplotypes_csv = working_dir / "haplotypes.csv"
    summary_csv = working_dir / "summary.csv"
    par_summary_csv = working_dir / "par_summary.csv"

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
                        intermediates_dir='{report.parent}', \
                        params=list( \
                            sample_id = '{sample_id}', \
                            input_bam = '{bam}', \
                            str_catalouge = '{str_catalouge}', \
                            reads_csv = '{reads_csv}', \
                            filtered_reads_csv = '{filtered_reads_csv}', \
                            consensus_csv = '{consensus_csv}', \
                            clustering_summary_csv = '{haplotypes_csv}', \
                            test_summary_csv = '{summary_csv}', \
                            par_summary_csv = '{par_summary_csv}' \
                        ) \
                    ) \
                    """,
        ],
        text=True,
        check=False,
        capture_output=True,
    )

    logger.debug("Rscript output: %s", process.stdout)
    logger.debug("Rscript error: %s", process.stderr)

    if process.returncode != 0:
        logger.error("Rscript failed with error code %d", process.returncode)
        raise typer.Exit(code=1)

    logger.info("Finished!")


if __name__ == "__main__":
    app()
