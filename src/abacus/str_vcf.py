from itertools import groupby
from pathlib import Path
from textwrap import dedent

from pyfaidx import Fasta

from abacus.config import config
from abacus.consensus import ConsensusCall
from abacus.locus import Locus
from abacus.parameter_estimation import HomozygousParameters
from abacus.utils import Haplotype


def generate_vcf_header(reference: Path, sample_name: str) -> str:
    # Start building the header
    header = "##fileformat=VCFv4.5\n##source=Abacus\n"

    # Handle reference and contig information
    header += f"##reference=file://{reference}\n"

    reference_index_file_path = reference.with_suffix(reference.suffix + ".fai")
    if not reference_index_file_path.exists():
        error_message = f"Reference index file not found: {reference_index_file_path}"
        raise FileNotFoundError(error_message)

    with reference_index_file_path.open() as f:
        for row in f:
            columns = row.strip().split("\t")
            contig_name, contig_size = columns[0], columns[1]
            header += f"##contig=<ID={contig_name},length={contig_size}>\n"

    # Add standard VCF headers; SVLEN, RN, RUS and RUC are the reserved VCFv4.5 tandem repeat keys
    header += dedent(
        """\
        ##FILTER=<ID=PASS,Description="All filters passed">
        ##FILTER=<ID=NoReads,Description="No call: no reads at the locus">
        ##FILTER=<ID=NoCall,Description="No reliable call could be produced">
        ##ALT=<ID=CNV:TR,Description="Tandem repeat">
        ##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Length of structural variant">
        ##INFO=<ID=RN,Number=A,Type=Integer,Description="Total number of repeat sequences in this allele">
        ##INFO=<ID=RUS,Number=.,Type=String,Description="Repeat unit sequence of the corresponding repeat sequence">
        ##INFO=<ID=RUC,Number=.,Type=Float,Description="Repeat unit count of corresponding repeat sequence">
        ##INFO=<ID=LPSRUS,Number=A,Type=String,Description="Repeat unit sequence of the longest pure segment in the allele">
        ##INFO=<ID=LPSRUC,Number=A,Type=Integer,Description="Repeat unit count of the longest pure segment in the allele">
        ##INFO=<ID=TRUC,Number=A,Type=Float,Description="Estimated total number of repeat units in the allele">
        ##INFO=<ID=CITRUC,Number=.,Type=Float,Description="Confidence interval around TRUC">
        ##INFO=<ID=REFDIV,Number=A,Type=Float,Description="Divergence of the allele's consensus from the catalog locus structure (0-1); a high value means the repeat counts don't describe the allele well">
        ##INFO=<ID=CONS,Number=A,Type=String,Description="Consensus sequence of the allele">
        ##INFO=<ID=LOCUSID,Number=1,Type=String,Description="Locus identifier">
        ##INFO=<ID=REPID,Number=1,Type=String,Description="Repeat identifier">
        ##INFO=<ID=STRSTATUS,Number=A,Type=String,Description="Status of TRUC against the catalog thresholds: normal (<= BENIGNMAX), intermediate or pathogenic (>= PATHOGENICMIN)">
        ##INFO=<ID=BENIGNMAX,Number=1,Type=Integer,Description="Largest benign repeat unit count, from the catalog">
        ##INFO=<ID=PATHOGENICMIN,Number=1,Type=Integer,Description="Smallest pathogenic repeat unit count, from the catalog">
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        ##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier; alleles sharing this ID are phased consistently (same H1/H2 order) across all satellite records of a locus">
        ##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
        ##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depth">
        ##FORMAT=<ID=ADSP,Number=R,Type=Integer,Description="Number of spanning reads consistent with the allele">
        ##FORMAT=<ID=ADFL,Number=R,Type=Integer,Description="Number of flanking reads consistent with the allele">\n""",
    )

    # Add column headers to the header
    header += f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}"

    return header


def _join(values) -> str:
    return ",".join(str(value) for value in values)


def _satellite_kmers(consensus_call: ConsensusCall, satellite_idx: int) -> list[str]:
    """Consensus kmers aligned to one satellite of the locus."""
    # Satellite and break nodes match the kmers one-to-one; sub-satellite nodes have no kmer of their own
    path = [node for node in consensus_call.alignment.path if ("sub_" in node or "satellite_" in node or "break" in node) and not node.startswith("sub_")]
    is_satellite = ["satellite" in node and int(node.removeprefix("satellite_").split("_")[0]) == satellite_idx for node in path]

    kmers = consensus_call.obs_kmer_string.split("|")
    return [kmer for kmer, keep in zip(kmers, is_satellite) if keep]


def create_vcf_records(
    consensus_calls: list[ConsensusCall],
    ref_path: Path,
    final_params: dict[Haplotype, HomozygousParameters],
    locus_is_het: bool,
) -> list[str]:
    """Create a VCF record for each satellite of a single STR locus."""
    ref = Fasta(ref_path)
    if not consensus_calls:
        return []

    locus = consensus_calls[0].locus
    if len({call.locus.id for call in consensus_calls}) != 1:
        error_message = "All consensus calls must be at the same locus"
        raise ValueError(error_message)

    # One ALT allele per haplotype (H1, H2 or HOM); a haplotype without a consensus call has no repeat sequences
    alt_haplotypes = [Haplotype.H1, Haplotype.H2] if locus_is_het else [Haplotype.HOM]
    calls_by_haplotype = {call.haplotype: call for call in consensus_calls}
    alt_calls = [calls_by_haplotype.get(haplotype) for haplotype in alt_haplotypes]
    alt_params = [final_params[haplotype] for haplotype in alt_haplotypes]

    # Depths are the same for every satellite and follow ALT order; a haplotype without a consensus call has depth 0
    spanning_depths = [call.spanning_reads if call else 0 for call in alt_calls]
    flanking_depths = [call.flanking_reads if call else 0 for call in alt_calls]
    depths = [sp + fl for sp, fl in zip(spanning_depths, flanking_depths, strict=True)]

    # H1/H2 order is consistent across all satellite records of a locus (see PS), so a heterozygous genotype is phased;
    # a homozygous locus has one diploid allele
    sample: dict[str, str] = {"GT": "1|2" if locus_is_het else "1/1"}
    if locus_is_het:
        sample["PS"] = str(locus.location.start)
    sample["DP"] = str(sum(depths))
    # AD, ADSP and ADFL include the REF allele, which no read is assigned to
    sample["AD"] = _join([0, *depths])
    sample["ADSP"] = _join([0, *spanning_depths])
    sample["ADFL"] = _join([0, *flanking_depths])

    records = []
    for i, satellite in enumerate(locus.satellites):
        kmers_per_allele = [_satellite_kmers(call, i) if call else [] for call in alt_calls]

        # Repeat sequences: runs of identical consensus kmers, so interruptions and motif changes get their own run
        runs_per_allele = [[(kmer, len(list(group))) for kmer, group in groupby(kmers)] for kmers in kmers_per_allele]

        # Estimated total repeat units, with the confidence interval as offsets from the estimate
        truc = [params.mean[i] for params in alt_params]
        citruc = [bound for params, est in zip(alt_params, truc) for bound in (params.mean_ci_low[i] - est, params.mean_ci_high[i] - est)]

        # SVLEN is the reference length; a repeat absent from the reference has SVLEN 1
        svlen = max(satellite.location.end - satellite.location.start, 1)

        info = {
            "SVLEN": _join(svlen for _ in alt_haplotypes),
            "RN": _join(len(runs) for runs in runs_per_allele),
        }
        # RUS/RUC hold one entry per repeat sequence, so they are left out when no allele has any
        if any(runs_per_allele):
            info["RUS"] = _join(kmer for runs in runs_per_allele for kmer, _ in runs)
            info["RUC"] = _join(count for runs in runs_per_allele for _, count in runs)
        # Longest pure segment: the longest run of any repeat unit, the first one on ties (max keeps the first)
        lps_per_allele = [max(runs, key=lambda run: run[1]) if runs else (".", ".") for runs in runs_per_allele]
        info["LPSRUS"] = _join(kmer for kmer, _ in lps_per_allele)
        info["LPSRUC"] = _join(count for _, count in lps_per_allele)
        info["TRUC"] = _join(round(est, 3) for est in truc)
        info["CITRUC"] = _join(round(bound, 3) for bound in citruc)
        info["REFDIV"] = _join(round(call.alignment.str_ref_divergence, 3) if call else "." for call in alt_calls)
        if config.add_consensus_to_vcf:
            info["CONS"] = _join("".join(kmers) if kmers else "." for kmers in kmers_per_allele)
        info["LOCUSID"] = locus.id
        info["REPID"] = satellite.id
        # Threshold fields are only written for satellites with thresholds in the catalog
        if satellite.benign_max is not None or satellite.pathogenic_min is not None:
            info["STRSTATUS"] = _join(satellite.status(est) for est in truc)
        if satellite.benign_max is not None:
            info["BENIGNMAX"] = str(satellite.benign_max)
        if satellite.pathogenic_min is not None:
            info["PATHOGENICMIN"] = str(satellite.pathogenic_min)

        fields = [
            # CHROM: the chromosome of the locus
            locus.location.chrom,
            # POS and REF: the padding base immediately preceding the repeat (1-based POS == 0-based start)
            str(satellite.location.start),
            # ID: a unique identifier for the satellite record
            f"Abacus.{locus.id}.{i + 1}",
            # REF: the reference base at the padding position
            str(ref[locus.location.chrom][satellite.location.start - 1]),
            # ALT: one entry per haplotype, all with the same CNV:TR type
            _join("<CNV:TR>" for _ in alt_haplotypes),
            # QUAL
            ".",
            # FILTER
            "PASS",
            # INFO
            ";".join(f"{key}={value}" for key, value in info.items()),
            # FORMAT
            ":".join(sample),
            # Sample values for FORMAT fields
            ":".join(sample.values()),
        ]
        records.append("\t".join(fields))

    return records


def create_no_call_records(locus: Locus, ref_path: Path, depth: int) -> list[str]:
    """Create a no-call VCF record for each satellite of a locus without a call."""
    ref = Fasta(ref_path)
    records = []
    for i, satellite in enumerate(locus.satellites):
        info = {"LOCUSID": locus.id, "REPID": satellite.id}
        if satellite.benign_max is not None:
            info["BENIGNMAX"] = str(satellite.benign_max)
        if satellite.pathogenic_min is not None:
            info["PATHOGENICMIN"] = str(satellite.pathogenic_min)

        fields = [
            locus.location.chrom,
            str(satellite.location.start),
            f"Abacus.{locus.id}.{i + 1}",
            str(ref[locus.location.chrom][satellite.location.start - 1]),
            # No ALT allele, so the per-allele INFO fields are left out
            ".",
            ".",
            "NoReads" if depth == 0 else "NoCall",
            ";".join(f"{key}={value}" for key, value in info.items()),
            "GT:DP",
            f"./.:{depth}",
        ]
        records.append("\t".join(fields))

    return records


def write_vcf(
    vcf: Path,
    vcf_records_tmp: Path,
    reference: Path,
    sample_id: str,
) -> None:
    """Write STR results to VCF format."""
    # Get chromosome order from reference index file
    reference_index_file_path = reference.with_suffix(reference.suffix + ".fai")
    chrom_order: dict[str, int] = {}
    with reference_index_file_path.open() as fai_file:
        for i, row in enumerate(fai_file):
            chrom = row.strip().split("\t")[0]
            chrom_order[chrom] = i

    # Read pre-computed records, sort by genomic position, write
    records = [line for line in vcf_records_tmp.read_text().splitlines() if line]
    records.sort(key=lambda r: (chrom_order.get(r.split("\t")[0], float("inf")), int(r.split("\t")[1])))

    with vcf.open("w") as vcf_file:
        header = generate_vcf_header(reference, sample_id)
        vcf_file.write(header + "\n")
        for record in records:
            vcf_file.write(record + "\n")
