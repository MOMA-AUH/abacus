from collections import defaultdict
from pathlib import Path
from textwrap import dedent

from pyfaidx import Fasta

from abacus.consensus import ConsensusCall


def generate_vcf_header(reference: Path, sample_name: str, unique_alts: list[int]) -> str:
    # Start building the header
    header = "##fileformat=VCFv4.2\n##source=Abacus\n"

    # Handle reference and contig information
    header += f"##file://reference={reference}\n"

    reference_index_file_path = reference.with_suffix(reference.suffix + ".fai")
    if not reference_index_file_path.exists():
        error_message = f"Reference index file not found: {reference_index_file_path}"
        raise FileNotFoundError(error_message)

    with reference_index_file_path.open() as f:
        for row in f:
            columns = row.strip().split("\t")
            contig_name, contig_size = columns[0], columns[1]
            header += f"##contig=<ID={contig_name},length={contig_size}>\n"

    # Add standard VCF headers
    header += dedent(
        """\
        ##FILTER=<ID=PASS,Description="All filters passed">
        ##FILTER=<ID=LowQual,Description="Low quality variant">
        ##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">
        ##INFO=<ID=REF,Number=1,Type=Integer,Description="Number of repeat units in the reference">
        ##INFO=<ID=RU,Number=1,Type=String,Description="Repeat unit sequence">
        ##INFO=<ID=LOCUSID,Number=1,Type=String,Description="Variant identifier">
        ##INFO=<ID=REPID,Number=1,Type=String,Description="Repeat identifier">
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        ##FORMAT=<ID=LC,Number=1,Type=Integer,Description="Locus coverage">
        ##FORMAT=<ID=REPCN,Number=A,Type=Float,Description="Number of repeat units spanned by the consensus allele">
        ##FORMAT=<ID=REPEST,Number=A,Type=Float,Description="Number of repeat units estimated by the EM algorithm">
        ##FORMAT=<ID=REPCI,Number=A,Type=String,Description="Confidence interval for REPEST">
        ##FORMAT=<ID=AD,Number=A,Type=Integer,Description="Allelic depth">
        ##FORMAT=<ID=ADSP,Number=A,Type=Integer,Description="Number of spanning reads consistent with the allele">
        ##FORMAT=<ID=ADFL,Number=A,Type=Integer,Description="Number of flanking reads consistent with the allele">\n""",
    )
    # Add STR-specific headers
    for alt in unique_alts:
        header += f"""##ALT=<ID=STR{alt},Description="Short Tandem Repeat with {alt} repeat units">\n"""

    # Add column headers to the header
    header += f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}"

    return header


def create_vcf_records(consensus_calls: list[ConsensusCall], ref_path: Path) -> list[str]:
    """Create a VCF record for a single STR locus."""
    # Load reference
    ref = Fasta(ref_path)
    # Check if there are any consensus calls
    if not consensus_calls:
        return []

    # Extract locus information
    locus = consensus_calls[0].locus

    # Check if all consensus calls are at the same locus
    if len({call.locus.id for call in consensus_calls}) != 1:
        error_message = "All consensus calls must be at the same locus"
        raise ValueError(error_message)

    # Extract locus and satellite information
    n_satellites = len(locus.satellites)

    # CHROM field
    chrom_field = locus.location.chrom

    records = []
    for i in range(n_satellites):
        # Extract information for each satellite
        satellite = locus.satellites[i]
        sat_idx = i + 1

        # POS field
        pos_field = str(satellite.location.start)

        # ID field
        id_field = f"Abacus.{locus.id}.{sat_idx}"

        # REF field
        ref_field = str(ref[chrom_field][satellite.location.start : satellite.location.end])

        # ALT field
        alt_counts = [call.satellite_count[i] for call in consensus_calls]
        alt_alleles = [f"<STR{count}>" for count in alt_counts]
        unique_alts = list(set(alt_alleles))
        alt_field = ",".join(unique_alts)

        # QUAL field
        qual_field = "."

        # FILTER field
        filter_field = "PASS"

        # INFO field
        info_fields = [
            f"END={locus.location.end}",
            f"LOCUSID={locus.id}",
            f"RU={satellite.sequence}",
            f"REPID={satellite.id}",
        ]
        info_field = ";".join(info_fields)

        # FORMAT field / values

        # GT field
        gt_indices = []
        for alt_allele in alt_alleles:
            if alt_allele == ref_field:
                gt_indices.append("0")
            else:
                gt_indices.append(str(unique_alts.index(alt_allele) + 1))
        gt_format = "/".join(sorted(gt_indices))

        # Calculate counts for LC AD fields
        alt_depths = [call.spanning_reads + call.flanking_reads for call in consensus_calls]
        alt_spanning_depths = [call.spanning_reads for call in consensus_calls]
        alt_flanking_depths = [call.flanking_reads for call in consensus_calls]

        lc_field = str(sum(alt_depths))
        ad_field = ",".join(str(depth) for depth in alt_depths)
        adsp_field = ",".join(str(depth) for depth in alt_spanning_depths)
        adfl_field = ",".join(str(depth) for depth in alt_flanking_depths)

        # Get repeat count estimates for each allele
        repcn_field = ",".join(str(count) for count in alt_counts)
        # TODO: Calculate proper estimates (use EM algorithm estimates)
        repest_field = ",".join(str(count) for count in alt_counts)
        # TODO: Calculate proper confidence intervals
        repci_format = f"{alt_counts[0]}-{alt_counts[0]}"

        # Construct FORMAT and FORMAT values
        format_field_headers: list[str] = [
            "GT",
            "LC",
            "REPCN",
            "REPEST",
            "REPCI",
            "AD",
            "ADSP",
            "ADFL",
        ]

        format_field_values: list[str] = [
            gt_format,
            lc_field,
            repcn_field,
            repest_field,
            repci_format,
            ad_field,
            adsp_field,
            adfl_field,
        ]

        format_field = ":".join(format_field_headers)
        format_value = ":".join(format_field_values)

        # Construct VCF record
        fields: list[str] = [
            chrom_field,
            pos_field,
            id_field,
            ref_field,
            alt_field,
            qual_field,
            filter_field,
            info_field,
            format_field,
            format_value,
        ]

        record = "\t".join(fields)

        records.append(record)

    return records


def write_vcf(
    vcf: Path,
    consensus_calls: list[ConsensusCall],
    sample_id: str,
    reference: Path,
) -> None:
    """Write STR results to VCF format."""
    # Get chromosome order from reference index file
    reference_index_file_path = reference.with_suffix(reference.suffix + ".fai")
    chrom_order = {}
    # Read chromosome order from reference index file
    with reference_index_file_path.open() as fai_file:
        for i, row in enumerate(fai_file):
            chrom = row.strip().split("\t")[0]
            chrom_order[chrom] = i

    with vcf.open("w") as vcf_file:
        # Write VCF header
        unique_counts = list({count for call in consensus_calls for count in call.satellite_count})
        header = generate_vcf_header(reference, sample_id, unique_counts)
        vcf_file.write(header + "\n")

        # Group consensus calls by locus
        consensus_calls_by_locus: dict[str, list[ConsensusCall]] = defaultdict(list)
        for call in consensus_calls:
            consensus_calls_by_locus[call.locus.id].append(call)

        # Sort consensus calls dict by locus (chrom based on reference order, start, end)
        consensus_calls_by_locus_sorted = dict(
            sorted(
                consensus_calls_by_locus.items(),
                key=lambda item: (
                    chrom_order.get(item[1][0].locus.location.chrom, float("inf")),  # Sort by reference order
                    item[1][0].locus.location.start,
                    item[1][0].locus.location.end,
                ),
            ),
        )

        # Create VCF records
        for _, calls in consensus_calls_by_locus_sorted.items():
            records = create_vcf_records(calls, reference)
            vcf_file.write("\n".join(records) + "\n")
