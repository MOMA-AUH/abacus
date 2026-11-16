from dataclasses import dataclass

import numpy as np
import pandas as pd

from abacus.allele_grouping import estimate_group_parameters, infer_allele_groups
from abacus.consensus import ConsensusCall, consensus_calls_by_haplotype, update_labels_based_on_consensus
from abacus.filtering import filter_low_qual_read_calls, filter_outlier_qual_read_calls, filter_unsupported_haplotypes
from abacus.graph import ReadCall
from abacus.logging import logger
from abacus.parameter_estimation import HeterozygousParameters, HomozygousParameters
from abacus.timing import timed
from abacus.utils import Haplotype


@dataclass
class HaplotypingResult:
    grouped_read_calls: list[ReadCall]
    removed_read_calls: list[ReadCall]
    consensus_calls: list[ConsensusCall]
    final_params: dict[Haplotype, HomozygousParameters]
    het_params: HeterozygousParameters
    hom_params: HomozygousParameters
    test_summary: pd.DataFrame


def haplotype_locus(read_calls: list[ReadCall], ploidy: int) -> HaplotypingResult:
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

    # Infer candidate allele groups before applying quality filters within each group.
    with timed("infer_allele_groups") as t:
        initial_grouped_read_calls, initial_grouping_outliers, _, _, _, _ = infer_allele_groups(
            read_calls=good_read_calls,
            ploidy=ploidy,
        )
        all_removed_read_calls.extend(initial_grouping_outliers)
        t.info = f"{len(initial_grouped_read_calls)} grouped, {len(initial_grouping_outliers)} removed"

    # Filter QC outliers per candidate allele group.
    with timed("filter_outlier_qual_read_calls") as t:
        good_read_calls, outlier_quality_read_calls = filter_outlier_qual_read_calls(read_calls=initial_grouped_read_calls)
        all_removed_read_calls.extend(outlier_quality_read_calls)
        t.info = f"{len(good_read_calls)} kept, {len(outlier_quality_read_calls)} removed"

    # Refit provisional groups and parameters, then refine against consensus.
    with timed("infer_allele_groups") as t:
        grouped_read_calls, grouping_outliers, het_params, hom_params, group_params, test_summary_res_df = infer_allele_groups(
            read_calls=good_read_calls,
            ploidy=ploidy,
        )
        all_removed_read_calls.extend(grouping_outliers)
        t.info = f"{len(grouped_read_calls)} grouped, {len(grouping_outliers)} removed"

    consensus_calls = _refine_haplotypes(grouped_read_calls, group_params)
    all_removed_read_calls.extend(read for read in grouped_read_calls if read.haplotype == Haplotype.OUTLIER)
    grouped_read_calls = [read for read in grouped_read_calls if read.haplotype != Haplotype.OUTLIER]
    return HaplotypingResult(
        grouped_read_calls=grouped_read_calls,
        removed_read_calls=all_removed_read_calls,
        consensus_calls=consensus_calls,
        final_params=group_params,
        het_params=het_params,
        hom_params=hom_params,
        test_summary=test_summary_res_df,
    )


def _consensus_lengths(group_params: dict[Haplotype, HomozygousParameters]) -> dict[Haplotype, list[int]]:
    """Round fitted satellite counts, skipping empty or NaN estimates for absent groups."""
    return {h: [int(round(x)) for x in p.mean] for h, p in group_params.items() if p.mean.size and not np.isnan(p.mean).any()}


def _refine_haplotypes(grouped_read_calls: list[ReadCall], group_params: dict[Haplotype, HomozygousParameters]) -> list[ConsensusCall]:
    """Refine candidate allele groups, estimates, and consensus into the final haplotype calls.

    Read labels and group_params are updated in place. Removed reads keep their outlier labels
    in the caller's list for reporting. Ties preserve the original haplotype.
    """
    with timed("consensus"):
        active_reads = list(grouped_read_calls)
        while True:
            original_labels = [read.haplotype for read in active_reads]
            active_reads, unsupported = filter_unsupported_haplotypes(active_reads)
            if unsupported or original_labels != [read.haplotype for read in active_reads]:
                group_params.update(estimate_group_parameters(active_reads))

            expected_haplotypes = {read.haplotype for read in active_reads}
            lengths = _consensus_lengths(group_params)
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
                group_params.update(estimate_group_parameters(active_reads))
                continue

            if original_labels == [read.haplotype for read in active_reads]:
                final_consensus = raw_consensus
            else:
                if spanning_moved:
                    group_params.update(estimate_group_parameters(active_reads))
                    lengths = _consensus_lengths(group_params)
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
