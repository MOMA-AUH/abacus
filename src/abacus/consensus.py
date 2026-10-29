from dataclasses import dataclass, field

import numpy as np
from Levenshtein import distance as levenshtein_distance

from abacus.consensus_search import KmerRead, find_consensus
from abacus.graph import AlignmentType, ConsensusAlignment, ReadCall, qual_to_char
from abacus.locus import Locus
from abacus.logging import logger
from abacus.parameter_estimation import HomozygousParameters
from abacus.timing import timed
from abacus.utils import Haplotype, compute_levenshtein_rate, trim_sequences_for_comparison

# A slot in a locus's structure: a break (index into locus.breaks) or a satellite (index into
# locus.satellites). Consensus is built independently per slot - see create_consensus_calls.
Slot = tuple[str, int]  # ("break" | "satellite", index)


@dataclass
class ConsensusCall(ReadCall):
    spanning_reads: int = 0
    flanking_reads: int = 0

    consensus_string: str = field(init=False)

    def __post_init__(self: "ConsensusCall") -> None:
        self.consensus_string = contract_kmer_string(self.obs_kmer_string)

    def to_dict(self) -> dict:
        return super().to_dict() | {
            "consensus_strings": self.consensus_string,
            "spanning_reads": self.spanning_reads,
            "flanking_reads": self.flanking_reads,
        }


def contract_kmer_string(kmer_string: str) -> str:
    if not kmer_string:
        return ""
    kmer_list = kmer_string.split("|")
    contracted_kmer = ""
    prev_kmer = kmer_list[0]
    count = 1
    for kmer in kmer_list[1:]:
        if kmer == prev_kmer:
            count += 1
        else:
            contracted_kmer += f"{count}({prev_kmer})-"
            prev_kmer = kmer
            count = 1
    contracted_kmer += f"{count}({prev_kmer})"
    return contracted_kmer


def build_consensus_for_locus(grouped_read_calls: list[ReadCall], final_params: dict[Haplotype, HomozygousParameters]) -> list[ConsensusCall]:
    """Build consensus, reassign flanking reads in place, and rebuild only if labels change.

    Target lengths stay fixed: only flanking reads move, while spanning reads dominate length.
    """
    with timed("consensus"):
        expected_haplotypes = {r.haplotype for r in grouped_read_calls}
        target_lengths = consensus_lengths(grouped_read_calls, final_params)
        raw_consensus_calls = consensus_calls_by_haplotype(grouped_read_calls, target_lengths)
        # A missing haplotype must not move flanking reads to the surviving one
        # or produce a complete genotype from only a partial consensus.
        if {call.haplotype for call in raw_consensus_calls} != expected_haplotypes:
            logger.warning("Incomplete raw consensus; emitting NoCall for the whole locus.")
            return []
        original_haplotypes = [r.haplotype for r in grouped_read_calls]
        update_flanking_labels_based_on_consensus(read_calls=grouped_read_calls, consensus_read_calls=raw_consensus_calls)
        if original_haplotypes == [r.haplotype for r in grouped_read_calls]:
            final_consensus_calls = raw_consensus_calls
        else:
            final_consensus_calls = consensus_calls_by_haplotype(grouped_read_calls, target_lengths)
        if {call.haplotype for call in final_consensus_calls} != expected_haplotypes:
            logger.warning("Incomplete final consensus; emitting NoCall for the whole locus.")
            return []
        for call in final_consensus_calls:
            if call.spanning_reads == 0:
                logger.warning(
                    f"Locus {call.locus.id}, haplotype {call.haplotype}: consensus length has no spanning-read "
                    f"support ({call.flanking_reads} flanking reads only).",
                )
        return final_consensus_calls


def consensus_lengths(read_calls: list[ReadCall], final_params: dict[Haplotype, HomozygousParameters]) -> dict[Haplotype, list[int]]:
    """Per-satellite kmer count of each haplotype's consensus, from the estimated per-satellite
    means. A break isn't part of this - it always contributes exactly one kmer when the locus
    defines one, independent of any estimate (see locus_slots). Haplotypes without reads have NaN
    estimates and no consensus, so they're skipped.
    """
    return {h: [int(round(x)) for x in p.mean] for h, p in final_params.items() if p.mean.size and not np.isnan(p.mean).any()}


def consensus_calls_by_haplotype(read_calls: list[ReadCall], target_lengths: dict[Haplotype, list[int]]) -> list[ConsensusCall]:
    consensus_calls: list[ConsensusCall] = []
    for haplotype in {r.haplotype for r in read_calls}:
        haplotyped_read_calls = [r for r in read_calls if r.haplotype == haplotype]
        consensus_calls.extend(create_consensus_calls(read_calls=haplotyped_read_calls, haplotype=haplotype, target_lengths=target_lengths[haplotype]))
    return consensus_calls


def locus_slots(breaks: list[str], n_satellites: int) -> list[Slot]:
    """The locus's structure as an ordered list of slots: a break slot before satellite i whenever
    locus.breaks[i] is non-empty, then satellite i itself, plus a trailing break slot if the last
    entry in locus.breaks is non-empty.
    """
    slots: list[Slot] = []
    for i in range(n_satellites):
        if breaks[i]:
            slots.append(("break", i))
        slots.append(("satellite", i))
    if breaks[-1]:
        slots.append(("break", n_satellites))
    return slots


def segment_read_kmers(read_call: ReadCall, breaks: list[str]) -> list[tuple[Slot, list[str]]]:
    """Split a read's kmer string into the slots its own mapping reached, in locus order.

    satellite_count already says how many kmers of each satellite this read's own alignment
    covers (0 for one it never reached), and get_kmer_string lays a read's kmers out in locus
    order - so a slot's kmers are simply "however many are left to take" at that point. A
    left-flanking or spanning read's kmers are consumed from the front (its own alignment starts
    at the true left edge); a right-flanking read's kmers are a suffix of the locus, so they're
    consumed from the back and the result flipped back into locus order.
    """
    kmers = read_call.obs_kmer_string.split("|") if read_call.obs_kmer_string else []
    slots = locus_slots(breaks, len(read_call.satellite_count))

    is_right = read_call.alignment.type == AlignmentType.RIGHT_FLANKING
    if is_right:
        slots = slots[::-1]
        kmers = kmers[::-1]

    segments: list[tuple[Slot, list[str]]] = []
    pos = 0
    for slot in slots:
        kind, idx = slot
        if kind == "break":
            # get_kmer_string can append a spurious empty kmer for a break this read's own
            # alignment never actually reached (an out-of-range slice on its synced bases) - an
            # empty string is never real break content, so treat it as absent.
            if pos >= len(kmers) or kmers[pos] == "":
                continue
            segment = [kmers[pos]]
            pos += 1
        else:
            n = read_call.satellite_count[idx]
            segment = kmers[pos : pos + n]
            pos += n
        if segment:
            segments.append((slot, segment))

    if is_right:
        segments = [(slot, segment[::-1]) for slot, segment in reversed(segments)]

    return segments


def segment_reads_for_slot(read_calls: list[ReadCall], segmented: list[list[tuple[Slot, list[str]]]], slot: Slot) -> list[KmerRead]:
    """Reads covering one slot, with the "spanning"/"left"/"right" type find_consensus needs to weigh
    an edge as a true anchor or as merely where the read's own alignment stops.

    A spanning read is anchored at both true ends for every slot it covers. A flanking read is
    only ever boundary-uncertain at the one edge its own alignment doesn't reach past - its first
    slot for a right-flanking read, its last for a left-flanking one. Every other slot it covers
    is fully bounded by its neighbours in the read, so it behaves like a spanning read there.
    """
    reads: list[KmerRead] = []
    for read_call, segments in zip(read_calls, segmented, strict=True):
        matching = [i for i, (s, _) in enumerate(segments) if s == slot]
        if not matching:
            continue
        i = matching[0]
        alignment_type = read_call.alignment.type
        if alignment_type == AlignmentType.SPANNING:
            rtype = "spanning"
        elif alignment_type == AlignmentType.LEFT_FLANKING:
            rtype = "left" if i == len(segments) - 1 else "spanning"
        else:
            rtype = "right" if i == 0 else "spanning"
        reads.append((segments[i][1], rtype, read_call.alignment.strand))
    return reads


def build_consensus_call(
    locus: Locus,
    consensus_kmers: list[str],
    target_lengths: list[int],
    haplotype: Haplotype,
    spanning_reads: int,
    flanking_reads: int,
) -> ConsensusCall:
    """Construct a ConsensusCall directly from consensus kmers — no minigraph."""
    consensus_sequence = "".join(consensus_kmers)
    full_sequence = locus.left_anchor + consensus_sequence + locus.right_anchor
    sequence_length = len(full_sequence)

    # Build the graph path and reference kmers in one pass over the slot structure
    path = ["left_anchor_overlap"]
    ref_kmers: list[str] = []
    for kind, idx in locus_slots(locus.breaks, len(target_lengths)):
        if kind == "break":
            path.append(f"break_{idx}")
            ref_kmers.append(locus.breaks[idx])
        else:
            path.extend([f"satellite_{idx}"] * target_lengths[idx])
            ref_kmers.extend([locus.satellites[idx].sequences[0]] * target_lengths[idx])
    path.append("right_anchor_overlap")

    str_ref_divergence = compute_levenshtein_rate(consensus_sequence, "".join(ref_kmers))

    alignment = ConsensusAlignment(
        name=f"consensus_{haplotype}",
        sequence=full_sequence,
        qualities=[60] * sequence_length,
        mod_5mc_probs="0" * sequence_length,
        strand="+",
        locus=locus,
        n_soft_clipped_left=0,
        n_soft_clipped_right=0,
        str_sequence=consensus_sequence,
        type=AlignmentType.SPANNING,
        path=path,
        str_ref_divergence=str_ref_divergence,
        str_qualities=[60] * len(consensus_sequence),
        mean_str_quality=60.0,
        q10_str_quality=60,
    )

    return ConsensusCall(
        locus=locus,
        alignment=alignment,
        satellite_count=list(target_lengths),
        str_error_rate=0.0,
        obs_kmer_string="|".join(consensus_kmers),
        ref_kmer_string="|".join(ref_kmers),
        mod_5mc_kmer_string="|".join("0" * len(kmer) for kmer in consensus_kmers),
        qual_kmer_string="|".join(qual_to_char(60) * len(kmer) for kmer in consensus_kmers),
        haplotype=haplotype,
        spanning_reads=spanning_reads,
        flanking_reads=flanking_reads,
    )


def create_consensus_calls(read_calls: list[ReadCall], haplotype: Haplotype, target_lengths: list[int]) -> list[ConsensusCall]:
    """Build the haplotype's consensus by searching each satellite and break independently, then
    concatenating - each read is aligned against the slot the mapping actually placed its kmers
    in, not a flat position in the whole allele, so a satellite whose count differs between reads
    can't shift where a later satellite or break appears to sit.

    The consensus is constructed directly from the kmers and slot structure — no minigraph
    remapping. Slots with no read coverage default to the reference sequence.

    This also folds what used to be a separate flanking-only path into the same mechanism: with
    no spanning reads at all, left- and right-flanking reads simply contribute to whichever slots
    their own alignments reach, and one ordinary (SPANNING-shaped) consensus comes out as long as
    the slots are jointly covered. The result's `spanning_reads` says whether any of that coverage
    was a real spanning read, for callers that need to know the length estimate has no spanning
    support.
    """
    locus = read_calls[0].alignment.locus
    breaks = locus.breaks

    spanning_count = sum(1 for r in read_calls if r.alignment.type == AlignmentType.SPANNING)
    flanking_count = len(read_calls) - spanning_count

    segmented = [segment_read_kmers(r, breaks) for r in read_calls]

    consensus_kmers: list[str] = []
    for slot in locus_slots(breaks, len(target_lengths)):
        kind, idx = slot
        length = 1 if kind == "break" else target_lengths[idx]
        reads_for_slot = segment_reads_for_slot(read_calls, segmented, slot)
        slot_kmers = find_consensus(reads_for_slot, length, trim_boundaries=(kind == "satellite"))
        # No reads covered this slot — fill with N's (unknown bases)
        if len(slot_kmers) < length:
            unit_len = len(locus.breaks[idx]) if kind == "break" else len(locus.satellites[idx].sequences[0])
            slot_kmers.extend(["N" * unit_len] * (length - len(slot_kmers)))
        consensus_kmers.extend(slot_kmers)

    consensus_sequence = "".join(consensus_kmers)
    # Empty is only a call for a 0-copy allele; otherwise no read covered the locus
    if not consensus_sequence and any(target_lengths):
        return []

    return [build_consensus_call(locus, consensus_kmers, target_lengths, haplotype, spanning_count, flanking_count)]


def update_flanking_labels_based_on_consensus(
    read_calls: list[ReadCall],
    consensus_read_calls: list[ConsensusCall],
) -> None:
    # Get unique consensus haplotypes
    unique_haplotypes = {read_call.haplotype for read_call in consensus_read_calls}
    for read_call in read_calls:
        # Skip if read call is spanning
        if read_call.alignment.type == AlignmentType.SPANNING:
            continue

        # Find closest consensus and use this as haplotype
        # Initialize
        current_haplotype = read_call.haplotype
        closest_consensus_haplotype = Haplotype.NONE
        dist_to_closest = np.inf
        for haplotype in unique_haplotypes:
            # Get consensus read calls for this haplotype
            haplotype_consensus_read_calls = [x for x in consensus_read_calls if x.haplotype == haplotype]

            # Get group probabilities using string distance
            dist_to_consensus = calc_dist_to_consensus(
                read_call=read_call,
                consensus_read_calls=haplotype_consensus_read_calls,
            )

            # Check if this is the closest consensus (or tied for closest, in which case prefer current haplotype to avoid unnecessary label changes)
            if dist_to_consensus < dist_to_closest or (dist_to_closest == dist_to_consensus and haplotype == current_haplotype):
                dist_to_closest = dist_to_consensus
                closest_consensus_haplotype = haplotype

        read_call.set_haplotype(closest_consensus_haplotype)


def calc_dist_to_consensus(
    read_call: ReadCall,
    consensus_read_calls: list[ConsensusCall],
) -> float:
    # Extract information from reads
    seq = read_call.alignment.str_sequence
    seq_type = read_call.alignment.type

    best_dist = np.inf
    for consensus_read_call in consensus_read_calls:
        consensus_seq = consensus_read_call.alignment.str_sequence
        consensus_type = consensus_read_call.alignment.type

        # Skip if both are flanking and on different sides
        if AlignmentType.SPANNING not in (seq_type, consensus_type) and seq_type != consensus_type:
            continue

        seq_trimmed, consensus_seq_trimmed = trim_sequences_for_comparison(seq, seq_type, consensus_seq, consensus_type)
        dist = levenshtein_distance(seq_trimmed, consensus_seq_trimmed)

        # Check if distance is better than best distance
        best_dist = min(best_dist, dist)

    return best_dist
