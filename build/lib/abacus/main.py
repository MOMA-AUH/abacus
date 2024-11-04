import subprocess
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from pyfaidx import Fasta
from typing_extensions import Annotated

from abacus.config import config
from abacus.graph import (
    GraphAlignment,
    GroupedReadCall,
    get_read_calls,
    get_reads_in_locus,
    graph_align_reads_to_locus,
    group_flanking_read_calls,
    remap_flanking_alignments_to_locus,
)
from abacus.haplotyping import filter_read_calls, group_read_calls
from abacus.locus import load_loci_from_json
from abacus.logging import logger, set_log_file_handler

# Set up the CLI
app = typer.Typer()


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
    loci_subset: Optional[List[str]] = typer.Option(
        None,
        "--loci-subset",
        help="Subset of loci to process. If not provided, all loci will be processed. Use multiple times to specify multiple loci.",
    ),
    anchor_length: Annotated[
        int,
        typer.Option(
            "--anchor-length",
            help="Length of the left and right anchor sequences",
        ),
    ] = 500,
    min_anchor_overlap: Annotated[
        int,
        typer.Option(
            "--min-anchor-overlap",
            help="Minimum overlap between read and anchor",
        ),
    ] = 200,
    min_qual: Annotated[
        int,
        typer.Option(
            "--min-qual",
            help="Minimum median base quality in STR region",
        ),
    ] = 10,
) -> None:
    # Setup logging to file
    set_log_file_handler(logger, log_file)

    # Setup configuration
    config.ANCHOR_LEN = anchor_length
    config.MIN_ANCHOR_OVERLAP = min_anchor_overlap
    config.MIN_STR_READ_QUAL = min_qual

    # Welcome message
    logger.info("Running Abacus")

    # Load reference FASTA
    ref_fasta = Fasta(ref)

    # Load loci data from JSON
    loci = load_loci_from_json(str_catalouge, ref_fasta)

    # Check if loci subset is valid
    if loci_subset and set(loci_subset).isdisjoint([locus.id for locus in loci]):
        logger.error("No loci in subset found in STR catalouge")
        raise typer.Exit(code=1)

    # Initialize output dataframes
    all_read_calls: List[GroupedReadCall] = []
    all_filtered_reads: List[GraphAlignment] = []
    all_haplotyping = []
    all_summaries = []

    # Process each locus
    for locus in loci:
        # Skip loci not in subset
        if loci_subset and locus.id not in loci_subset:
            continue

        print(f"Processing {locus.id} {locus.structure}...")

        if not locus.satellites:
            print("No valid satellite pattern found in STR definition")
            continue

        # Get reads in locus
        reads = get_reads_in_locus(bam, locus)

        # Call STR in individual reads
        # TODO: Handle unmapped reads
        spanning_alignments, flanking_alignments, filtered_alignments, unmapped_reads = graph_align_reads_to_locus(reads, locus)

        # Handle spanning reads
        read_calls = get_read_calls(spanning_alignments, locus)

        # Filter read calls
        filtered_read_calls, good_read_calls = filter_read_calls(read_calls=read_calls)

        # Run EM algorithm
        haplotyping_res_df, test_summary_res_df, grouped_read_calls, h1_satellite_counts, h2_satellite_counts = group_read_calls(
            read_calls=good_read_calls, kmer_dim=len(locus.satellites)
        )

        # Group flanking reads
        # TODO: Handle unhandled reads
        # Re-map flanking reads
        remapped_flanking_reads, unhandled_1 = remap_flanking_alignments_to_locus(flanking_alignments, locus)

        good_remapped_flanking_reads = [aln for aln in remapped_flanking_reads if not aln.get_error_flags()]
        filtered_remapped_flanking_reads = [aln for aln in remapped_flanking_reads if aln.get_error_flags()]

        # Handle flanking reads
        flanking_read_calls = get_read_calls(good_remapped_flanking_reads, locus)

        # Filter flanking reads
        unhandled_3, good_flanking_read_calls = filter_read_calls(read_calls=flanking_read_calls)
        # Group flanking reads
        grouped_flanking_read_calls = group_flanking_read_calls(good_flanking_read_calls)

        # Combine results
        filtered_alignments.extend(filtered_remapped_flanking_reads)
        grouped_read_calls.extend(filtered_read_calls)
        grouped_read_calls.extend(grouped_flanking_read_calls)

        # TODO: Remove this
        # Annotate results with locus info
        for df in [haplotyping_res_df, test_summary_res_df]:
            for k, v in locus.to_dict().items():
                # Add locus info as a column
                df[k] = v

        satellite_df_list = [
            pd.DataFrame(
                {
                    "idx": sat_idx,
                    "satellite": locus.satellites[sat_idx].sequence,
                },
                index=[0],
            )
            for sat_idx in range(len(locus.satellites))
        ]
        satellite_df = pd.concat(satellite_df_list)

        haplotyping_res_df = pd.merge(haplotyping_res_df, satellite_df, on="idx", how="left")

        # Concatenate results
        all_read_calls.extend(grouped_read_calls)
        all_filtered_reads.extend(filtered_alignments)

        all_haplotyping.append(haplotyping_res_df)
        all_summaries.append(test_summary_res_df)

    # Create output directory
    output_dir = report.parent / "abacus_output"
    output_dir.mkdir(exist_ok=True)

    # Write output files
    reads_csv = output_dir / "reads.csv"
    filtered_reads_csv = output_dir / "filtered_reads.csv"
    haplotypes_csv = output_dir / "haplotypes.csv"
    summary_csv = output_dir / "summary.csv"

    with open(reads_csv, "w", encoding="utf-8") as f:
        pd.DataFrame([r.to_dict() for r in all_read_calls]).to_csv(f, index=False)
    with open(filtered_reads_csv, "w", encoding="utf-8") as f:
        pd.DataFrame([r.to_dict() for r in all_filtered_reads]).to_csv(f, index=False)
    with open(haplotypes_csv, "w", encoding="utf-8") as f:
        pd.concat(all_haplotyping).to_csv(f, index=False)
    with open(summary_csv, "w", encoding="utf-8") as f:
        pd.concat(all_summaries).to_csv(f, index=False)

    # Render report
    print("Render report...")

    report_name = report.name
    report_dir = report.parent

    report_template = Path(__file__).parent / "scripts" / "report_template.Rmd"

    logger.info("Render report")

    with open(log_file, "a") as f:
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
                            clustering_summary_csv = '{haplotypes_csv}', \
                            test_summary_csv = '{summary_csv}' \
                        ))""",
            ],
            text=True,
            check=True,
            stdout=f,
            stderr=f,
        )

    logger.info("Abacus finished")


if __name__ == "__main__":
    app()
