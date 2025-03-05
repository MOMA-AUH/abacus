import array
import random

import pysam
import pytest

from abacus.config import config
from abacus.graph import Locus, Read, get_graph_alignments, get_kmer_string, get_satellite_counts_from_path
from abacus.locus import process_str_pattern


@pytest.mark.parametrize(
    "structure, read, expected_satellite_counts",
    [
        # Single satellite tests
        pytest.param(
            "(AGA)+",
            "AGA",
            [1],
            id="Single satellite x 1",
        ),
        pytest.param(
            "(CAG)+",
            "CAG" * 10,
            [10],
            id="Single satellite x 10",
        ),
        pytest.param(
            "(TTA)+",
            "TTA" * 100,
            [100],
            id="Single satellite x 100",
        ),
        # Multiple satellites tests
        pytest.param(
            "(AGA)+(CAG)+",
            "AGA" * 3 + "CAG" * 5,
            [3, 5],
            id="Two satellites x (3, 5)",
        ),
        # Ambiguous bases
        # N = A or C or G or T
        # R = A or G
        # Y = C or T
        # W = A or T
        # K = G or T
        pytest.param(
            "(ANA)+",
            "AGA",
            [1],
            id="Ambiguity, N: x (1)",
        ),
        pytest.param(
            "(AGA)+(ANA)+",
            "AGA" * 5 + "ATA" * 3,
            [5, 3],
            id="Ambiguity, N: x (5 ,3)",
        ),
        pytest.param(
            "(N)+(AGA)+",
            "T" * 5 + "AGA",
            [5, 1],
            id="Ambiguity, N: x (5, 1)",
        ),
        pytest.param(
            "(N)+",
            "T" * 19,
            [19],
            id="Ambiguity, N: x (19)",
        ),
        pytest.param(
            "(ARA)+",
            "AGA",
            [1],
            id="Ambiguity, R: x (1)",
        ),
        pytest.param(
            "(ATA)+(ARA)+",
            "ATA" * 5 + "AGA" * 3,
            [5, 3],
            id="Ambiguity, R: x (5 ,3)",
        ),
        pytest.param(
            "(R)+(CAC)+(Y)+",
            "G" * 5 + "CAC" * 3 + "T" * 7,
            [5, 3, 7],
            id="Ambiguity, R,Y: x (5, 3, 7)",
        ),
        # Skippability
        pytest.param(
            "(TTC)*",
            "",
            [0],
            id="Skippability x 0",
        ),
        pytest.param(
            "(TTC)*",
            "TTC" * 17,
            [17],
            id="Skippability x 17",
        ),
        pytest.param(
            "(AGA)*(CAG)*",
            "",
            [0, 0],
            id="Skippability x (0, 0)",
        ),
        pytest.param(
            "(AGA)*(CAG)*",
            "AGA" * 11 + "CAG" * 13,
            [11, 13],
            id="Skippability x (11, 13)",
        ),
        pytest.param(
            "(AGA)*(CAG)*",
            "CAG",
            [0, 1],
            id="Skippability x (0, 1)",
        ),
        # Skipability with N
        pytest.param(
            "(N)*(AGA)*",
            "T" * 0 + "AGA" * 7,
            [0, 7],
            id="Skippability with N x (0, 7)",
        ),
        # Breaks
        pytest.param(
            "(AGA)*TTTTT(CAG)*",
            "AGA" * 0 + "TTTTT" + "CAG" * 0,
            [0, 0],
            id="Breaks: Internal x (0, 0)",
        ),
        pytest.param(
            "(AGA)*TTTTT(CAG)*",
            "AGA" * 7 + "TTTTT" + "CAG" * 0,
            [7, 0],
            id="Breaks: Internal x (7, 0)",
        ),
        pytest.param(
            "(AGA)*TTTTT(CAG)*",
            "AGA" * 0 + "TTTTT" + "CAG" * 5,
            [0, 5],
            id="Breaks: Internal x (0, 5)",
        ),
        pytest.param(
            "(AGA)*TTTTT(CAG)*",
            "AGA" * 7 + "TTTTT" + "CAG" * 5,
            [7, 5],
            id="Breaks: Internal x (7, 5)",
        ),
        pytest.param(
            "AAC(AAG)*(CAG)*AGG",
            "AAC" + "AAG" * 11 + "CAG" * 13 + "AGG",
            [11, 13],
            id="Breaks: Pre and post x (11, 13)",
        ),
        pytest.param(
            "AAC(AAG)*TTT(CAG)*AGG",
            "AAC" + "AAG" * 3 + "TTT" + "CAG" * 2 + "AGG",
            [3, 2],
            id="Breaks: Pre, internal and post x (3, 2)",
        ),
        # Edge cases
        pytest.param(
            "(AGA)+(CAG)*",
            "AGA" * 0 + "CAG" * 3,
            [1, 2],
            id="Edge case: Forced mismatch with +",
        ),
        pytest.param(
            "(TTTTTTTTTC)+",
            "AAAAAAAAAC" * 29,
            [29],
            id="Edge case: High percentage of mismatches in satellite",
        ),
        pytest.param(
            "(NTN)+(TNT)+",
            "ATG" * 3 + "TAT" * 17,
            [3, 17],
            id="Edge case: High percentage of Ns in satellites",
        ),
        pytest.param(
            "(AGN)*",
            "AGT" * 100,
            [100],
            id="Edge case: Many copies",
        ),
    ],
)
def test_get_satellite_counts_from_path(structure, read, expected_satellite_counts):
    alphabet = "ATCG"

    # Set seed for reproducibility
    random.seed(42)

    # Make random left and right anchors
    left_anchor = "".join(random.choices(alphabet, k=config.anchor_len))
    right_anchor = "".join(random.choices(alphabet, k=config.anchor_len))

    read_str = f"{left_anchor}{read}{right_anchor}"

    read_id = "read"

    reads = [
        Read(
            name=read_id,
            sequence=read_str,
            qualities=[30] * len(read_str),
            strand="+",
            mod_5mc_probs="",
            locus_id="locus1",
        ),
    ]

    satellites, breaks = process_str_pattern(structure)

    locus = Locus(
        id="locus1",
        chrom="chr1",
        start=100,
        end=200,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
        structure=structure,
        satellites=satellites,
        breaks=breaks,
    )

    graph_aligments = get_graph_alignments(reads, locus)
    # TODO: Fix this error
    graph_aligment = next(a for a in graph_aligments if a.name == read_id)
    path = graph_aligment.path
    satellite_counts = get_satellite_counts_from_path(locus=locus, path=path)

    assert satellite_counts == expected_satellite_counts


@pytest.mark.parametrize(
    "structure, read, expected_expected_kmer_string, expected_observed_kmer_string",
    [
        pytest.param(
            "(CAG)*",
            "",
            "",
            "",
            id="Empty read",
        ),
        pytest.param(
            "(CAG)*",
            "CAG" * 10,
            "-".join(["CAG"] * 10),
            "-".join(["CAG"] * 10),
            id="Single satellite x 10",
        ),
        pytest.param(
            "(CAG)*",
            "CAG" * 4 + "TTT" + "CAG" * 5,
            "-".join(["CAG"] * 10),
            "-".join(["CAG"] * 4 + ["TTT"] + ["CAG"] * 5),
            id="Single w error satellite x 10",
        ),
    ],
)
def test_get_satellite_strings(structure, read, expected_expected_kmer_string, expected_observed_kmer_string):
    alphabet = "ATCG"

    # Set seed for reproducibility
    random.seed(42)

    # Make random left and right anchors
    left_anchor = "".join(random.choices(alphabet, k=config.anchor_len))
    right_anchor = "".join(random.choices(alphabet, k=config.anchor_len))

    read_str = f"{left_anchor}{read}{right_anchor}"

    read_id = "read"

    reads = [
        Read(
            name=read_id,
            sequence=read_str,
            qualities=[30] * len(read_str),
            strand="+",
            mod_5mc_probs="",
            locus_id="locus1",
        ),
    ]

    satellites, breaks = process_str_pattern(structure)

    locus = Locus(
        id="locus1",
        chrom="chr1",
        start=100,
        end=200,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
        structure=structure,
        satellites=satellites,
        breaks=breaks,
    )

    graph_aligments = get_graph_alignments(reads=reads, locus=locus)
    graph_aligment = next(a for a in graph_aligments if a.name == read_id)

    satellite_counts = get_satellite_counts_from_path(locus=locus, path=graph_aligment.path)

    observed_kmer_string, expected_kmer_string = get_kmer_string(
        locus=locus,
        satellite_counts=satellite_counts,
        synced_list=graph_aligment.str_sequence_synced,
    )

    assert expected_kmer_string == expected_expected_kmer_string
    assert observed_kmer_string == expected_observed_kmer_string


def create_aligned_segment(query_name: str, query_sequence: str, mm_tag: str, ml_tag: list[int]) -> pysam.AlignedSegment:
    a = pysam.AlignedSegment()
    a.query_name = query_name
    a.query_sequence = query_sequence

    # Methylation tags
    a.set_tag("MM", mm_tag)
    a.set_tag("ML", array.array("B", ml_tag))
    return a


@pytest.mark.parametrize(
    ("query_name", "query_sequence", "mm_tag", "ml_tag", "expected_methylation"),
    [
        pytest.param(
            "read1",
            "AAAA",
            "C+m?,;",
            [],
            [0.0, 0.0, 0.0, 0.0],
            id="No methylation",
        ),
        pytest.param(
            "read2",
            "C",
            "C+m?,0;",
            [255],
            [1.0],
            id="With one methylated C",
        ),
        pytest.param(
            "read2",
            "ACTTTTTCCAACCCTAACTCGTTCAGTTGCGTATTGCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCAACCCCCACCCTCACCCTCACCCTCACCCTCACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCTAACCCCTAACCCCTAACCCTAACCCTAACCCCTAACCCCTAACCCCTAACCCCTAACCCTAACCCTAACCCTAACCCAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTCTAACCCTCTAACCCTAACCCTAACCCTCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTACCCTAACCCTACCCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTACCCTAACCCCAACCCCAACCCCAACCCCAACCCCAACCCCAACCCTAACCCTAACCCTAACCCTAACCCTACCCTAACCCTAACCCTAACCCTAA",
            "C+h?,7,1;C+m?,7,1;",
            [32, 6, 56, 11],
            [0],
            id="Test2",
        ),
        pytest.param(
            "read3",
            "ATCG",
            "C+m,0;",
            [128],
            [0.5, 0.0, 0.0, 0.0],
            id="With partial methylation",
        ),
        pytest.param(
            "read4",
            "ATCGATCG",
            "C+m,0,4;",
            [128, 64],
            [0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
            id="With multiple methylation",
        ),
    ],
)
def test_methylation_from_alignment(
    query_name,
    query_sequence,
    mm_tag,
    ml_tag,
    expected_methylation,
):
    alignment = create_aligned_segment(
        query_name=query_name,
        query_sequence=query_sequence,
        mm_tag=mm_tag,
        ml_tag=ml_tag,
    )
    read = Read.from_aligment(alignment)
    assert read.mod_5mc_probs == expected_methylation


# TODO: Fix these test cases
# @pytest.mark.parametrize(
#     "id, structure, chr, start, end, bam_path_str",
#     [
#         pytest.param(
#             "test",
#             "(CGG)*",
#             "chrX",
#             147912050,
#             147912110,
#             "/faststorage/project/MomaDiagnosticsHg38/NO_BACKUP/nanopore/nanopore-methylation-pipeline/N575/N575_Repeat_Expansion_Validation_DEVEL/output/106241996890_N575_008/106241996890_N575_008_alignment_phased.bam",
#             id="Case 1: Low QUAL read e.g. 75276889-3579-4198-b254-6f2157573234",
#         ),
#         pytest.param(
#             "test",
#             "(CAGG)*(CAGA)*(CA)*",
#             "chr3",
#             129172576,
#             129172732,
#             "/faststorage/project/MomaDiagnosticsHg38/NO_BACKUP/nanopore/nanopore-methylation-pipeline/N575/N575_Repeat_Expansion_Validation_DEVEL/output/106241996890_N575_008/106241996890_N575_008_alignment_phased.bam",
#             id="Case 2: Low QUAL read e.g. 7e320817-fa55-49ee-a946-0cd0b6629cc4	",
#         ),
#         pytest.param(
#             "EIF4A3",
#             "(CCTCGCTGCGCCGCTGCCGA)*(CCTCGCTGTGCCGCTGCCGA)*",
#             "chr17",
#             80147022,
#             80147139,
#             "/faststorage/project/MomaDiagnosticsHg38/NO_BACKUP/nanopore/nanopore-methylation-pipeline/N575/N575_Repeat_Expansion_Validation_DEVEL/output/106284154866_N575_002/106284154866_N575_002_alignment_phased.bam",
#             id="Case 3: Stop at EIF4A3",
#         ),
#     ],
# )
# def test_cases(id: str, structure: str, chr: str, start: int, end: int, bam_path_str: str):
#     ref = Fasta("/faststorage/project/MomaReference/BACKUP/hg38/reference_genome/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna")

#     satellites, breaks = process_str_pattern(structure)

#     left_anchor = str(ref[chr][(start - config.ANCHOR_LEN) : start])
#     right_anchor = str(ref[chr][end : (end + config.ANCHOR_LEN)])

#     locus = Locus(
#         id=id,
#         chrom=chr,
#         start=start,
#         end=end,
#         left_anchor=left_anchor,
#         right_anchor=right_anchor,
#         structure=structure,
#         satellites=satellites,
#         breaks=breaks,
#     )

#     read_calls, _ = get_reads_in_locus(bam=Path(bam_path_str), locus=locus)

#     haplotyping_res_df, all_read_calls, test_summary_res_df = filter_read_calls(read_calls=read_calls)

#     pd.DataFrame([r.to_dict() for r in all_read_calls])

#     file = "/faststorage/project/MomaNanoporeDevelopment/BACKUP/devel/simond/abacus/testing/test_graph.csv"

#     pd.DataFrame([r.to_dict() for r in all_read_calls]).to_csv(file, index=False)

#     assert True
