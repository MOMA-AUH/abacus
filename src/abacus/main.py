import subprocess
from pathlib import Path

import pandas as pd
import pysam
import typer
from pyfaidx import Fasta
from typing_extensions import Annotated

from abacus.config import config
from abacus.graph import process_reads_in_str_region
from abacus.haplotyping import call_haplotypes
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
    config_file: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to the configuration file (JSON)",
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
    read_info: Annotated[
        Path,
        typer.Option(
            "--read-info",
            help="Path to the output CSV file of individual read info",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ],
    filtered_reads_info: Annotated[
        Path,
        typer.Option(
            "--filtered-reads-info",
            help="Path to the output CSV file of filtered reads",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ],
    haplotype_info: Annotated[
        Path,
        typer.Option(
            "--haplotype-info",
            help="Path to the output CSV file of haplotype calls",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ],
    summary: Annotated[
        Path,
        typer.Option(
            "--summary",
            help="Path to the output CSV file of summary statistics",
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
            help="Path to the log file",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ],
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
    ] = 15,
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
    loci = load_loci_from_json(config_file, ref_fasta)

    # Open BAM file
    bamfile = pysam.AlignmentFile(str(bam), "rb")

    # Initialize output dataframes
    haplotyping_df = pd.DataFrame()
    read_calls_df = pd.DataFrame()
    filtered_reads_df = pd.DataFrame()
    summary_df = pd.DataFrame()

    # Process each locus
    for locus in loci:
        # if locus.id not in ["AR", "HTT", "RFC1_alt", "ATXN8OS", "CNBP"]:
        # if locus.id not in ["AR", "HTT", "CNBP", "FMR1", "FGF14", "DMPK"]:
        # if locus.id not in ["FGF14"]:
        if locus.id not in ["ATXN1", "FGF14", "FGF14_alt", "HTT"]:
            continue

        print(f"Processing {locus.id} {locus.structure}...")

        if not locus.satellites:
            print("No valid satellite pattern found in STR definition")
            continue

        # Call STR in individual reads
        read_calls, filtered_reads = process_reads_in_str_region(bamfile=bamfile, locus=locus)

        filtered_reads_res_df = pd.DataFrame([r.to_dict() for r in filtered_reads])

        # Call STR haplotypes
        haplotyping_res_df, read_calls_res_df, test_summary_res_df = call_haplotypes(read_calls=read_calls)

        # TODO: Remove this
        # Annotate results with locus info
        for df in [haplotyping_res_df, test_summary_res_df]:
            for k, v in locus.to_dict().items():
                df[k] = v

        # Add satellite information to haplotype results satellite = locus.satellites[idx]
        satellite_df = pd.DataFrame()
        for i in range(len(locus.satellites)):
            satellite_df = pd.concat(
                [
                    satellite_df,
                    pd.DataFrame({"satellite": locus.satellites[i].sequence, "em_haplotype": f"h{i+1}"}, index=[0]),
                ],
                ignore_index=True,
                axis=0,
            )

        haplotyping_res_df = pd.merge(haplotyping_res_df, satellite_df, on="em_haplotype", how="left")

        # Concatenate results
        haplotyping_df = pd.concat([haplotyping_df, haplotyping_res_df], axis=0, ignore_index=True)
        read_calls_df = pd.concat([read_calls_df, read_calls_res_df], axis=0, ignore_index=True)
        summary_df = pd.concat([summary_df, test_summary_res_df], axis=0, ignore_index=True)
        filtered_reads_df = pd.concat([filtered_reads_df, filtered_reads_res_df], axis=0, ignore_index=True)

    # Write output
    with open(read_info, "w", encoding="utf-8") as f:
        read_calls_df.to_csv(f, index=False)
    with open(filtered_reads_info, "w", encoding="utf-8") as f:
        filtered_reads_df.to_csv(f, index=False)
    with open(haplotype_info, "w", encoding="utf-8") as f:
        haplotyping_df.to_csv(f, index=False)
    with open(summary, "w", encoding="utf-8") as f:
        summary_df.to_csv(f, index=False)

    # Render report
    print("Render report...")

    report_name = report.stem
    report_dir = report.parent

    report_template = "/faststorage/project/MomaNanoporeDevelopment/BACKUP/devel/simond/abacus/report.Rmd"

    process = subprocess.run(
        [
            "Rscript",
            "-e",
            f"""
                rmarkdown::render('{report_template}', \
                    output_file='{report_name}', \
                    output_dir='{report_dir}', \
                    intermediates_dir='{report_dir}', \
                    params=list( \
                        reads_csv = '{read_info}', \
                        filtered_reads_csv = '{filtered_reads_info}', \
                        clustering_summary_csv = '{haplotype_info}', \
                        test_summary_csv = '{summary}' \
                    ))""",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # Log stdout and stderr
    logger.info("Render report stdout:")
    logger.info(process.stdout)
    logger.info("Render report stderr:")
    logger.info(process.stderr)

    logger.info("Abacus finished")


if __name__ == "__main__":
    app()
