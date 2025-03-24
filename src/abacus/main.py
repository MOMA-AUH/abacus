from __future__ import annotations

import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from abacus.config import config
from abacus.consensus import ConsensusCall, create_consensus_calls
from abacus.graph import (
    FilteredRead,
    ReadCall,
    get_read_calls,
)
from abacus.group_summary import calculate_final_group_summaries
from abacus.haplotyping import filter_read_calls, group_read_calls
from abacus.locus import load_loci_from_json
from abacus.logging import logger, set_log_file_handler
from abacus.preprocess import get_reads_in_locus
from abacus.str_vcf import write_vcf


class Sex(StrEnum):
    XX = "XX"
    XY = "XY"


# Set up the CLI
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def abacus(
    # Inputs
    bam: Annotated[
        Path,
        typer.Option(
            "--bam",
            "-i",
            help="Path to the input BAM file",
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
            help="Path to the reference genome FASTA file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    sample_id: Annotated[
        str,
        typer.Option(
            "--sample-id",
            "-n",
            help="Sample ID",
        ),
    ],
    str_catalouge: Annotated[
        Path,
        typer.Option(
            "--str-catalouge",
            "-s",
            help="Path to the STR catalouge JSON file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    report: Annotated[
        Path,
        typer.Option(
            "--report",
            "-o",
            help="Path to the output HTML report",
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
            help="Path to the output VCF file",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ],
    log_file: Annotated[
        Path,
        typer.Option(
            "--log-file",
            "-l",
            help="Path to the log file",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ],
    loci_subset: Annotated[
        list[str] | None,
        typer.Option(
            "--loci-subset",
            help="Subset of loci to process. If not provided, all loci will be processed. Use multiple times to specify multiple loci.",
        ),
    ] = None,
    sex: Annotated[
        Sex,
        typer.Option(
            "--sex",
            help="Sex of the sample.",
            case_sensitive=False,
        ),
    ] = Sex.XX,
    # Configuration
    anchor_length: Annotated[
        int,
        typer.Option(
            "--anchor-length",
            help="Length of the left and right anchor sequences",
            rich_help_panel="Configuration Options",
        ),
    ] = config.anchor_len,
    min_anchor_overlap: Annotated[
        int,
        typer.Option(
            "--min-anchor-overlap",
            help="Minimum overlap between read and anchor",
            rich_help_panel="Configuration Options",
        ),
    ] = config.min_anchor_overlap,
    min_qual: Annotated[
        int,
        typer.Option(
            "--min-qual",
            help="Minimum median base quality in STR region",
            rich_help_panel="Configuration Options",
        ),
    ] = config.min_str_read_qual,
    max_trim: Annotated[
        int,
        typer.Option(
            "--max-trim",
            help="Maximum number of bases to trim from read ends",
            rich_help_panel="Configuration Options",
        ),
    ] = config.max_trim,
    error_rate_threshold: Annotated[
        float,
        typer.Option(
            "--error-rate-threshold",
            help="Threshold for error rate",
            rich_help_panel="Configuration Options",
        ),
    ] = config.error_rate_threshold,
) -> None:
    # TODO: Update logs with welcome, and loci messages
    # Setup logging to file
    set_log_file_handler(logger, log_file)

    # Setup configuration
    config.anchor_len = anchor_length
    config.min_anchor_overlap = min_anchor_overlap
    config.min_str_read_qual = min_qual
    config.max_trim = max_trim
    config.error_rate_threshold = error_rate_threshold

    # Welcome message
    logger.info("Running Abacus")

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
    for locus in loci:
        # Skip loci not in subset
        if loci_subset and locus.id not in loci_subset:
            continue

        # TODO: Take into account the sex of the sample
        print(f"Processing {locus.id} {locus.structure}...")

        if not locus.satellites:
            print("No valid satellite pattern found in STR definition")
            continue

        # Get reads in locus
        reads = get_reads_in_locus(bam, locus)

        # Call STR in individual reads
        read_calls, unmapped_reads = get_read_calls(reads, locus)

        # Filter read calls
        # TODO: Go through filters and clean up unused and unnecessary filters
        filtered_read_calls, good_read_calls = filter_read_calls(read_calls=read_calls)

        # TODO: Use optimize for homozygous model
        # TODO: Min N (eg 2) reads should be required to call a haplotype - if less, then call as homozygous

        # Get ploidy
        if locus.location.chrom == "chrY":
            ploidy = sex.value.count("Y")
        elif locus.location.chrom == "chrX":
            ploidy = sex.value.count("X")
        else:
            ploidy = 2

        # Group read calls
        grouped_read_calls, test_summary_res_df, par_summary_df = group_read_calls(
            read_calls=good_read_calls,
            ploidy=ploidy,
        )

        # Calculate final group summaries
        haplotyping_df = calculate_final_group_summaries(grouped_read_calls)

        # Create consensus for each haplotype
        # TODO: Use external tool for assembly
        # TODO: Use flanking for consensus
        # TODO: Use ONLY flanking for consensus
        unique_haplotypes = {r.em_haplotype for r in grouped_read_calls}
        for haplotype in unique_haplotypes:
            haplotyped_read_calls = [r for r in grouped_read_calls if r.em_haplotype == haplotype]
            consensus_sequences = create_consensus_calls(read_calls=haplotyped_read_calls, haplotype=haplotype)
            all_consensus_calls.extend(consensus_sequences)

        # Combine results
        grouped_read_calls.extend(filtered_read_calls)

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
        par_summary_df = par_summary_df.merge(satellite_df, on="idx", how="left")

        # Concatenate results
        all_read_calls.extend(grouped_read_calls)
        all_filtered_reads.extend(unmapped_reads)

        all_haplotyping_df.append(haplotyping_df)
        all_summaries_df.append(test_summary_res_df)
        all_par_summaries_df.append(par_summary_df)

    # Create output directory
    output_dir = report.parent / "abacus_output"
    output_dir.mkdir(exist_ok=True)

    # Write output files
    reads_csv = output_dir / "reads.csv"
    filtered_reads_csv = output_dir / "filtered_reads.csv"
    consensus_csv = output_dir / "consensus.csv"

    haplotypes_csv = output_dir / "haplotypes.csv"
    summary_csv = output_dir / "summary.csv"
    par_summary_csv = output_dir / "par_summary.csv"

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
    print("Render report...")
    report_name = report.name
    report_dir = report.parent

    report_template = Path(__file__).parent / "scripts" / "report_template.Rmd"

    logger.info("Render report")
    # TODO: Strømlin faver på tvæts af Abacus plots for samme locus
    with Path.open(log_file, "a") as f:
        subprocess.run(
            [
                "Rscript",
                "-e",
                f"""
                    rmarkdown::render('{report_template}', \
                        output_file='{report_name}', \
                        output_dir='{report_dir}', \
                        intermediates_dir='{report_dir}', \
                        params=list( \
                            sample_id = '{sample_id}', \
                            input_bam = '{bam}', \
                            str_catalouge = '{str_catalouge}', \
                            reads_csv = '{reads_csv}', \
                            filtered_reads_csv = '{filtered_reads_csv}', \
                            consensus_csv = '{consensus_csv}', \
                            clustering_summary_csv = '{haplotypes_csv}', \
                            em_summary_csv = '{summary_csv}', \
                            par_summary_csv = '{par_summary_csv}' \
                        ) \
                    ) \
                    """,
            ],
            text=True,
            check=True,
            stdout=f,
            stderr=f,
        )

    logger.info("Abacus finished")


if __name__ == "__main__":
    app()
