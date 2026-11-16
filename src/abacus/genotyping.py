from dataclasses import dataclass

import numpy as np
import pandas as pd

from abacus.consensus import ConsensusCall, consensus_calls_by_haplotype, update_labels_based_on_consensus
from abacus.filtering import filter_low_qual_read_calls, filter_outlier_qual_read_calls, filter_unsupported_haplotypes
from abacus.graph import ReadCall
from abacus.haplotyping import _estimate_final_parameters, run_haplotyping
from abacus.logging import logger
from abacus.parameter_estimation import HeterozygousParameters, HomozygousParameters
from abacus.timing import timed
from abacus.utils import Haplotype


@dataclass
class GenotypingResult:
    grouped_read_calls: list[ReadCall]
    removed_read_calls: list[ReadCall]
    consensus_calls: list[ConsensusCall]
    final_params: dict[Haplotype, HomozygousParameters]
    het_params: HeterozygousParameters
    hom_params: HomozygousParameters
    test_summary: pd.DataFrame


def genotype_locus(read_calls: list[ReadCall], ploidy: int) -> GenotypingResult:
    """Filter mapped calls and infer supported haplotypes and consensus without changing input calls.

    Heterozygous/homozygous test parameters describe the pre-consensus model comparison;
    final parameters describe the groups after consensus refinement.
    """
    read_calls = [read.copy() for read in read_calls]
    all_removed_read_calls: list[ReadCall] = []

    # Prefilter low quality read calls
    with timed("filter_low_qual_read_calls") as t:
        good_read_calls, low_quality_read_calls = filter_low_qual_read_calls(read_calls=read_calls)
        all_removed_read_calls.extend(low_quality_read_calls)
        t.info = f"{len(good_read_calls)} kept, {len(low_quality_read_calls)} removed"

    # Group reads before applying quality filters within each haplotype.
    with timed("run_haplotyping") as t:
        initial_grouped_read_calls, initial_haplotyping_outliers, _, _, _, _ = run_haplotyping(
            read_calls=good_read_calls,
            ploidy=ploidy,
        )
        all_removed_read_calls.extend(initial_haplotyping_outliers)
        t.info = f"{len(initial_grouped_read_calls)} grouped, {len(initial_haplotyping_outliers)} removed"

    # Filter QC outliers per haplotype group
    with timed("filter_outlier_qual_read_calls") as t:
        good_read_calls, outlier_quality_read_calls = filter_outlier_qual_read_calls(read_calls=initial_grouped_read_calls)
        all_removed_read_calls.extend(outlier_quality_read_calls)
        t.info = f"{len(good_read_calls)} kept, {len(outlier_quality_read_calls)} removed"

    # Refit after quality filtering, then refine assignments against consensus.
    with timed("run_haplotyping") as t:
        grouped_read_calls, haplotyping_outliers, het_params, hom_params, final_params, test_summary_res_df = run_haplotyping(
            read_calls=good_read_calls,
            ploidy=ploidy,
        )
        all_removed_read_calls.extend(haplotyping_outliers)
        t.info = f"{len(grouped_read_calls)} grouped, {len(haplotyping_outliers)} removed"

    consensus_calls = _refine_haplotypes_and_consensus(grouped_read_calls, final_params)
    all_removed_read_calls.extend(read for read in grouped_read_calls if read.haplotype == Haplotype.OUTLIER)
    grouped_read_calls = [read for read in grouped_read_calls if read.haplotype != Haplotype.OUTLIER]
    return GenotypingResult(
        grouped_read_calls=grouped_read_calls,
        removed_read_calls=all_removed_read_calls,
        consensus_calls=consensus_calls,
        final_params=final_params,
        het_params=het_params,
        hom_params=hom_params,
        test_summary=test_summary_res_df,
    )


def _consensus_lengths(final_params: dict[Haplotype, HomozygousParameters]) -> dict[Haplotype, list[int]]:
    """Round fitted satellite counts, skipping empty or NaN estimates for absent groups."""
    return {h: [int(round(x)) for x in p.mean] for h, p in final_params.items() if p.mean.size and not np.isnan(p.mean).any()}


def _refine_haplotypes_and_consensus(grouped_read_calls: list[ReadCall], final_params: dict[Haplotype, HomozygousParameters]) -> list[ConsensusCall]:
    """Refine read assignments and consensus, enforcing support and refitting final parameters.

    Read labels and final_params are updated in place. Removed reads keep their outlier labels
    in the caller's list for reporting. Ties preserve the original haplotype.
    """
    with timed("consensus"):
        active_reads = list(grouped_read_calls)
        while True:
            original_labels = [read.haplotype for read in active_reads]
            active_reads, unsupported = filter_unsupported_haplotypes(active_reads)
            if unsupported or original_labels != [read.haplotype for read in active_reads]:
                final_params.update(_estimate_final_parameters(active_reads))

            expected_haplotypes = {read.haplotype for read in active_reads}
            lengths = _consensus_lengths(final_params)
            raw_consensus = consensus_calls_by_haplotype(active_reads, lengths)
            if {call.haplotype for call in raw_consensus} != expected_haplotypes:
                logger.warning("Incomplete raw consensus; emitting NoCall for the whole locus.")
                return []

            original_labels = [read.haplotype for read in active_reads]
            spanning_labels = [read.haplotype for read in active_reads if read.is_spanning()]
            update_labels_based_on_consensus(active_reads, raw_consensus)
            spanning_moved = spanning_labels != [read.haplotype for read in active_reads if read.is_spanning()]
            active_reads, unsupported = filter_unsupported_haplotypes(active_reads)
            if unsupported or {read.haplotype for read in active_reads} != expected_haplotypes:
                final_params.update(_estimate_final_parameters(active_reads))
                continue

            if original_labels == [read.haplotype for read in active_reads]:
                final_consensus = raw_consensus
            else:
                if spanning_moved:
                    final_params.update(_estimate_final_parameters(active_reads))
                    lengths = _consensus_lengths(final_params)
                final_consensus = consensus_calls_by_haplotype(active_reads, lengths)
            if {call.haplotype for call in final_consensus} != expected_haplotypes:
                logger.warning("Incomplete final consensus; emitting NoCall for the whole locus.")
                return []
            for call in final_consensus:
                if call.spanning_reads == 0:
                    logger.warning(
                        f"Locus {call.locus.id}, haplotype {call.haplotype}: consensus length has no spanning-read "
                        f"support ({call.flanking_reads} flanking reads only).",
                    )
            return final_consensus
