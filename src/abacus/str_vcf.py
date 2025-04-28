from pathlib import Path
from textwrap import dedent

from pyfaidx import Fasta

from abacus.config import config
from abacus.consensus import ConsensusCall, contract_kmer_string
from abacus.locus import Locus
from abacus.parameter_estimation import HeterozygousParameters, HomozygousParameters
from abacus.utils import Haplotype


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
        ##FORMAT=<ID=REPCI,Number=A,Type=String,Description="Confidence interval for REPCN">
        ##FORMAT=<ID=AD,Number=A,Type=Integer,Description="Allelic depth">
        ##FORMAT=<ID=ADSP,Number=A,Type=Integer,Description="Number of spanning reads consistent with the allele">
        ##FORMAT=<ID=ADFL,Number=A,Type=Integer,Description="Number of flanking reads consistent with the allele">
        ##FORMAT=<ID=CONS,Number=A,Type=String,Description="Consensus sequence">
        ##FORMAT=<ID=CONTRCONS,Number=A,Type=String,Description="Contracted consensus sequence">\n""",
    )
    # Add STR-specific headers
    for alt in unique_alts:
        header += f"""##ALT=<ID=STR{alt},Description="Short Tandem Repeat with {alt} repeat units">\n"""

    # Add column headers to the header
    header += f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}"

    return header


def create_vcf_records(
    consensus_calls: list[ConsensusCall],
    ref_path: Path,
    het_params: HeterozygousParameters,
    hom_params: HomozygousParameters,
    locus_is_het: bool,
) -> list[str]:
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
        # In case of insertion (START == END), use the base at the start position
        if not ref_field:
            ref_field = str(ref[chrom_field][satellite.location.start])

        # ALT field
        repcn_est = [het_params.mean_h1[i], het_params.mean_h2[i]] if locus_is_het else [hom_params.mean[i]]
        alt_alleles = [f"<STR{int(count)}>" for count in repcn_est]
        alt_field = ",".join(alt_alleles)

        # QUAL field
        qual_field = "."

        # FILTER field
        filter_field = "PASS"

        # INFO field
        info_fields = [
            f"END={satellite.location.end}",
            f"LOCUSID={locus.id}",
            f"RU={satellite.sequence}",
            f"REPID={satellite.id}",
        ]
        info_field = ";".join(info_fields)

        # FORMAT field / values
        # Initialize
        format_field_headers: list[str] = []
        format_field_values: list[str] = []

        # GT field
        gt_indices = [str(i + 1) for i in range(len(alt_alleles))]
        gt_format = "/".join(gt_indices)

        # Add to format field
        format_field_headers.append("GT")
        format_field_values.append(gt_format)

        # Calculate counts for LC AD fields
        alt_depths = [call.spanning_reads + call.flanking_reads for call in consensus_calls]
        alt_spanning_depths = [call.spanning_reads for call in consensus_calls]
        alt_flanking_depths = [call.flanking_reads for call in consensus_calls]

        lc_field = str(sum(alt_depths))
        ad_field = ",".join(str(depth) for depth in alt_depths)
        adsp_field = ",".join(str(depth) for depth in alt_spanning_depths)
        adfl_field = ",".join(str(depth) for depth in alt_flanking_depths)

        # Add to format field
        format_field_headers.extend(["LC", "AD", "ADSP", "ADFL"])
        format_field_values.extend([lc_field, ad_field, adsp_field, adfl_field])

        # Repeat counts and confidence intervals
        # Get repeat confidence intervals
        repcn_est_low = [het_params.mean_h1_ci_low[i], het_params.mean_h2_ci_low[i]] if locus_is_het else [hom_params.mean_ci_low[i]]
        repcn_est_high = [het_params.mean_h1_ci_high[i], het_params.mean_h2_ci_high[i]] if locus_is_het else [hom_params.mean_ci_high[i]]

        repcn_field = ",".join(str(round(count, 3)) for count in repcn_est)
        repci_format = ",".join(f"{round(low, 3)}-{round(high, 3)}" for low, high in zip(repcn_est_low, repcn_est_high))

        # Add to format field
        format_field_headers.extend(["REPCN", "REPCI"])
        format_field_values.extend([repcn_field, repci_format])

        if config.add_consensus_to_vcf or config.add_contracted_consensus_to_vcf:
            # Consensus call fields
            subset_kmer_lists: list[list[str]] = []

            # Filter and organize the consensus calls
            if locus_is_het:
                filtered_consensus_calls = [call for call in consensus_calls if call.haplotype in [Haplotype.H1, Haplotype.H2]]
                filtered_consensus_calls.sort(key=lambda call: call.haplotype)
            else:
                filtered_consensus_calls = [call for call in consensus_calls if call.haplotype == Haplotype.HOM]

            # If more than one consensus call for a haplotype, use the longest one
            for haplotype in [Haplotype.H1, Haplotype.H2, Haplotype.HOM]:
                filtered_calls = [call for call in filtered_consensus_calls if call.haplotype == haplotype]
                if len(filtered_calls) > 1:
                    # Sort by length and take the longest one
                    filtered_calls.sort(key=lambda call: len(call.obs_kmer_string), reverse=True)
                    filtered_consensus_calls = [filtered_calls[0]]

            # Extract the kmer strings for each consensus call
            for consensus_call in filtered_consensus_calls:
                # Extract the path and extract the relevant nodes
                path = consensus_call.alignment.path
                path = [node for node in path if "sub_" in node or "satellite_" in node or "break" in node]
                # Find the index of the current satellite in the path
                index_list = ["satellite" in node and int(node.removeprefix("sub_").removeprefix("satellite_").split("_")[0]) == i for node in path]
                # Filter the kmer list based on the satellite index list
                kmer_list = consensus_call.obs_kmer_string.split("-")
                subset_kmer_lists.append([kmer_list[idx] for idx, is_satellite in enumerate(index_list) if is_satellite])

            # Consensus string
            if config.add_consensus_to_vcf:
                satellite_consensus_list = ["".join(x) for x in subset_kmer_lists]
                consensus_field = ",".join(satellite_consensus_list)

                # Add to format field
                format_field_headers.append("CONS")
                format_field_values.append(consensus_field)

            if config.add_contracted_consensus_to_vcf:
                # Contracted consensus call fields
                contracted_consensus_fields = [contract_kmer_string("-".join(kmer_list)) for kmer_list in subset_kmer_lists]
                contracted_consensus_field = ",".join(contracted_consensus_fields)

                # Add to format field
                format_field_headers.append("CONTRCONS")
                format_field_values.append(contracted_consensus_field)

        # Finalize format field
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
    het_params_dict: dict[str, HeterozygousParameters],
    hom_params_dict: dict[str, HomozygousParameters],
    locus_is_het_dict: dict[str, bool],
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
        unique_loci: list[Locus] = []
        for call in consensus_calls:
            if call.locus not in unique_loci:
                unique_loci.append(call.locus)

        # Sort locus by (chrom based on reference order, start, end)
        sorted_loci = sorted(
            unique_loci,
            key=lambda locus: (
                chrom_order.get(locus.location.chrom, float("inf")),  # Sort by reference order
                locus.location.start,
                locus.location.end,
            ),
        )

        # Create VCF records
        for locus in sorted_loci:
            # Get the locus ID
            locus_id = locus.id
            # Get all consensus calls for the current locus
            calls = [call for call in consensus_calls if call.locus.id == locus_id]
            # Get the parameters and heterozygosity status for the current locus
            het_params = het_params_dict.get(locus_id)
            hom_params = hom_params_dict.get(locus_id)
            locus_is_het = locus_is_het_dict.get(locus_id)
            # Check if any parameters are missing
            if het_params is None or hom_params is None or locus_is_het is None:
                error_message = f"Missing parameters for locus {locus_id}"
                raise ValueError(error_message)
            # Create VCF records for the current locus
            records = create_vcf_records(calls, reference, het_params, hom_params, locus_is_het)
            vcf_file.write("\n".join(records) + "\n")
