# Fixed-length consensus search for a repeat-unit sequence, used by consensus.py.
#
# Reads are kmer lists (one kmer per repeat unit), each with a type ("spanning", or "left"/"right"
# flanking, covering the consensus from one end) and a strand. Each distinct kmer is encoded as
# one character, so the search runs on strings. The length is given by the caller (haplotyping).
#
# 1. Initial consensus: a per-position vote at the target length (_initial_consensus).
# 2. Each iteration aligns every read to the consensus and counts what the reads have at each
#    position (_ColumnVotes). Reads far from the consensus (_voting_reads) don't vote, but are
#    still scored.
#    a. Strong changes - more than half of the covering reads, on every strand, agree on a change -
#       are applied all at once: substitutions, deletions and insertions, fitted back to the
#       length. A deletion plus an insertion elsewhere shifts everything between them. The round
#       is kept if it improves the objective.
#    b. Otherwise, disputed positions get scored substitutions and adjacent swaps. Each move is
#       rescored only within SCORE_WINDOW units of it, and all improving moves at least two
#       windows apart are applied together.
#
# Objective, compared in order: (summed OSA distance, composition penalty, summed squared
# distance). OSA counts a shifted interruption as one transposition, the same as flattening it
# (one substitution); composition (distance from the reads' median kmer counts) breaks that tie,
# and the squared distance prefers being a little off for every read over fitting one exactly.

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil, exp, lgamma, log
from statistics import median
from typing import NamedTuple

from rapidfuzz.distance import OSA, Opcodes
from rapidfuzz.distance import Levenshtein as RFLevenshtein

# Strand support for a kmer may come from this many positions either side of it
STRAND_MARGIN = 1
# Reads a strong change needs
MIN_READ_SUPPORT = 2
# A weak candidate needs at least this fraction of covering reads proposing the same change
MIN_SUPPORT_FRACTION = 0.05
# A read is an outlier only above this consensus error rate: the fraction of its kmers that differ
# from the consensus
MIN_OUTLIER_CONSENSUS_ERROR_RATE = 0.15
# A move is rescored only within this many units of it; moves applied together are two windows apart
SCORE_WINDOW = 40
# Log-probabilities below this underflow to 0.0 in exp()
MIN_LOG_PROBABILITY = -700


# One-char-per-kmer sequence, paired with "spanning" | "left" | "right", and strand "+" | "-"
EncodedRead = tuple[str, str, str]
# Raw kmer list, paired with "spanning" | "left" | "right", and strand "+" | "-"
KmerRead = tuple[list[str], str, str]
# (total OSA distance, composition penalty, summed squared distance)
Score = tuple[int, int, int]


class Move(NamedTuple):
    """A substitution of the kmer at `pos`, or a swap of the kmers at `pos` and `pos + 1`."""

    kind: str  # "sub" or "swap"
    pos: int
    kmer: str = ""  # The new kmer, for a substitution


class AlignedRead(NamedTuple):
    kmers: str
    offset: int  # Where the part of the consensus the read covers starts
    opcodes: Opcodes  # Alignment of the covered part to the read
    strand: str

    @property
    def covered_len(self) -> int:
        return self.opcodes[-1].src_end if self.opcodes else 0


ScoredMove = tuple[Score, Move]

# Printable, non-reserved unicode range for kmer encoding. This encoding never reaches spoa
# (unlike equal_length_split.py's _USABLE_KMER_CHARS), so it isn't limited to single-byte-safe ASCII.
_USABLE_CHARS = [chr(i) for i in range(33, 0x110000) if chr(i).isprintable()][:20000]


def find_consensus(reads: list[KmerRead], length: int, max_iters: int = 100, trim_boundaries: bool = True) -> list[str]:
    """Return the consensus kmer list of exactly `length` kmers for `reads`.

    Reads are (kmer list, "spanning"|"left"|"right" type, strand). Trims partial boundary kmers,
    encodes, runs the search, decodes.

    `trim_boundaries=False` skips the trim, for a locus break: it is one full kmer whenever a read
    reaches it, and with one read the trim would drop that kmer as uncorroborated.
    """
    if trim_boundaries:
        reads = _trim_partial_boundary_kmers(reads)
    encoded_reads, char_to_kmer = _encode_reads(reads)
    result = _find_consensus_encoded(encoded_reads, length, max_iters=max_iters)
    return [char_to_kmer[c] for c in result]


def _find_consensus_encoded(reads: list[EncodedRead], length: int, max_iters: int = 100) -> str:
    """Search for the consensus in single-char-per-kmer space; see find_consensus for the public API."""
    if length <= 0 or not reads:
        return ""
    target_counts = _target_composition(reads)

    # Each read's OSA distance to the part of the consensus it covers
    def distances_to(consensus: str) -> list[int]:
        return [OSA.distance(_covered_part(consensus, len(kmers), read_type)[0], kmers) for kmers, read_type, _ in reads]

    # Compared in order: see the module comment
    def objective(consensus: str, read_distances: list[int]) -> Score:
        return (sum(read_distances), _composition_penalty(consensus, target_counts), sum(d * d for d in read_distances))

    consensus = _initial_consensus(reads, length)
    for _ in range(max_iters):
        read_distances = distances_to(consensus)
        current_score = objective(consensus, read_distances)
        alignments = _align_reads(consensus, reads)
        # Outlier reads don't vote but are scored; by id, as reads can be equal
        voter_ids = {id(read) for read in _voting_reads(reads, read_distances)}
        voter_alignments = [alignment for alignment, read in zip(alignments, reads, strict=True) if id(read) in voter_ids]
        votes = _count_column_votes(consensus, voter_alignments)

        # 1. Every strong change at once, kept if it improves the objective
        update = _apply_strong_changes(consensus, votes, length)
        if update != consensus and objective(update, distances_to(update)) < current_score:
            consensus = update
            continue

        # 2. Otherwise the scored moves on the disputed positions that improve the objective
        update = _apply_weak_changes(
            consensus,
            votes,
            alignments=alignments,
            read_distances=read_distances,
            target_counts=target_counts,
            current_score=current_score,
        )
        if update != consensus:
            consensus = update
            continue

        # Neither step improved the consensus
        break
    return consensus


def _trim_partial_boundary_kmers(reads: list[KmerRead]) -> list[KmerRead]:
    """Drop a flanking read's boundary kmer unless another read has it as an interior kmer.

    The boundary kmer (last of a left-flanking read, first of a right-flanking one) may be a
    partial unit, cut where the read ends. Only interior kmers corroborate it, so two reads cut at
    the same point can't validate each other.
    """
    interior_kmers = {kmer for kmers, read_type, _ in reads for kmer in _interior(kmers, read_type)}
    trimmed = []
    for kmers, read_type, strand in reads:
        kept = kmers
        if read_type == "left" and kmers and kmers[-1] not in interior_kmers:
            kept = kmers[:-1]
        elif read_type == "right" and kmers and kmers[0] not in interior_kmers:
            kept = kmers[1:]
        if kept:
            trimmed.append((kept, read_type, strand))
    return trimmed


def _interior(kmers: list[str], read_type: str) -> list[str]:
    """Return a read's kmers without its boundary kmer: the last of a left-flanking read, the first of a right-flanking one."""
    if read_type == "left":
        return kmers[:-1]
    if read_type == "right":
        return kmers[1:]
    return kmers


def _encode_reads(reads: list[KmerRead]) -> tuple[list[EncodedRead], dict[str, str]]:
    """Return the reads with one character per kmer, and the map back from character to kmer."""
    distinct_kmers = sorted({k for kmers, *_ in reads for k in kmers})
    if len(distinct_kmers) > len(_USABLE_CHARS):
        msg = "too many distinct kmers for this quick encoding scheme"
        raise ValueError(msg)
    # Not strict: there are fewer kmers than characters, checked above
    kmer_to_char = dict(zip(distinct_kmers, _USABLE_CHARS, strict=False))
    encoded = [("".join(kmer_to_char[k] for k in kmers), read_type, strand) for kmers, read_type, strand in reads]
    return encoded, {v: k for k, v in kmer_to_char.items()}


def _initial_consensus(reads: list[EncodedRead], length: int) -> str:
    """Return the initial consensus: a weighted vote per position at the target length.

    A flanking read votes by offset from its anchored end. A spanning read k units off the target
    has k length errors somewhere along it, so consensus position p maps to its position p + m
    (p - m if short), with m ~ Binomial(k, p / (length - 1)) errors before p: its vote is spread
    over k + 1 positions, sliding from its left end to its right end along the consensus. Ties, and
    positions no read reaches, go to the most common kmer overall.

    A kmer only wins a position if every strand present voted for it there (within STRAND_MARGIN),
    so a strand-specific ONT error can't enter the consensus by majority.
    """
    kmer_totals = Counter(c for kmers, _, _ in reads for c in kmers)
    strands_present = {strand for _, _, strand in reads}
    position_votes: list[dict[str, float]] = [defaultdict(float) for _ in range(length)]
    strands_seen: dict[tuple[int, str], set[str]] = defaultdict(set)
    for kmers, read_type, strand in reads:
        n = len(kmers)
        # A spanning read's length error, in units
        k = abs(n - length)
        # Log-factorials up to k, shared by every position
        log_factorials = [lgamma(i + 1) for i in range(k + 1)] if read_type == "spanning" else []
        for pos in range(length):
            if read_type == "left":
                read_votes = [(kmers[pos], 1.0)] if pos < n else []
            elif read_type == "right":
                # Anchored at the far end: count back from it
                i = n - 1 - (length - 1 - pos)
                read_votes = [(kmers[i], 1.0)] if 0 <= i < n else []
            else:
                # Expected share of the k length errors before pos
                q = pos / (length - 1) if length > 1 else 0.5
                read_votes = []
                for m, weight in enumerate(_binomial_pmfs(k, q, log_factorials)):
                    # Errors before pos (m) shift the read position by m
                    i = pos + m if n >= length else pos - m
                    if 0 <= i < n:
                        read_votes.append((kmers[i], weight))
            # One vote per read per position, split over its possible kmers
            norm = sum(w for _, w in read_votes)
            for kmer, w in read_votes:
                position_votes[pos][kmer] += w / norm
                strands_seen[pos, kmer].add(strand)
    consensus = []
    for pos in range(length):
        # Only kmers seen on every strand may win, if any are
        eligible = {k: v for k, v in position_votes[pos].items() if _on_every_strand(strands_seen, pos, k, strands_present)} or position_votes[pos]
        if not eligible:
            consensus.append(kmer_totals.most_common(1)[0][0])
            continue
        # Ties go to the kmer most common overall
        best = max(eligible.values())
        consensus.append(max((k for k, v in eligible.items() if v == best), key=lambda k: kmer_totals[k]))
    return "".join(consensus)


def _binomial_pmfs(k: int, q: float, log_factorials: list[float]) -> list[float]:
    """Return P(m) for m = 0..k, with m ~ Binomial(k, q) and log_factorials[i] = log(i!).

    Computed in log-space: comb(k, m) alone overflows a float for a read thousands of kmers off
    the target length, while the full probability just underflows to 0.0.
    """
    # At the ends, all errors are after (q = 0) or before (q = 1) the position
    if q <= 0.0:
        return [1.0] + [0.0] * k
    if q >= 1.0:
        return [0.0] * k + [1.0]
    log_q, log_not_q = log(q), log(1 - q)
    pmfs = []
    for m in range(k + 1):
        log_pmf = log_factorials[k] - log_factorials[m] - log_factorials[k - m] + m * log_q + (k - m) * log_not_q
        pmfs.append(exp(log_pmf) if log_pmf > MIN_LOG_PROBABILITY else 0.0)
    return pmfs


def _align_reads(consensus: str, reads: list[EncodedRead]) -> list[AlignedRead]:
    """Align each read to the part of the consensus it covers."""
    alignments = []
    for kmers, read_type, strand in reads:
        covered, offset = _covered_part(consensus, len(kmers), read_type)
        opcodes = RFLevenshtein.opcodes(covered, kmers)
        alignments.append(AlignedRead(kmers, offset, opcodes, strand))
    return alignments


def _voting_reads(reads: list[EncodedRead], read_distances: list[int]) -> list[EncodedRead]:
    """Return the reads that aren't outliers (median + 3.5 MAD) in consensus error rate."""
    consensus_error_rates = [distance / max(len(kmers), 1) for distance, (kmers, _, _) in zip(read_distances, reads, strict=True)]
    med = median(consensus_error_rates)
    mad = median(abs(rate - med) for rate in consensus_error_rates)
    # Floor, so a MAD of 0 doesn't make every imperfect read an outlier
    max_consensus_error_rate = max(med + 3.5 * mad, MIN_OUTLIER_CONSENSUS_ERROR_RATE)
    return [read for read, rate in zip(reads, consensus_error_rates, strict=True) if rate <= max_consensus_error_rate]


@dataclass
class _ColumnVotes:
    """What the reads have at each consensus position, read off their alignments to the consensus."""

    n_covering: list[int]  # Reads covering each position; the extra last entry is the end
    observed: list[Counter[str]]  # What they have there: a kmer, or "" for a gap
    n_inserting: list[int]  # Reads inserting kmers just before each position
    insertions: list[Counter[str]]  # What they insert there
    strands_seen: dict[tuple[int, str], set[str]]  # Strands showing each (position, kmer), or (position, "+" + insertion)
    strands_present: set[str]

    def add_read(self, read: AlignedRead) -> None:
        kmers, offset, opcodes, strand = read
        for op in opcodes:
            # One read kmer per consensus position
            if op.tag in ("equal", "replace"):
                for d in range(op.src_end - op.src_start):
                    pos, kmer = offset + op.src_start + d, kmers[op.dest_start + d]
                    self.observed[pos][kmer] += 1
                    self.strands_seen[pos, kmer].add(strand)
            # Positions the read lacks
            elif op.tag == "delete":
                for pos in range(offset + op.src_start, offset + op.src_end):
                    self.observed[pos][""] += 1
                    self.strands_seen[pos, ""].add(strand)
            # Kmers between positions, counted at the position after them
            elif op.tag == "insert":
                pos, block = offset + op.src_start, kmers[op.dest_start : op.dest_end]
                self.n_inserting[pos] += 1
                self.insertions[pos][block] += 1
                # Strands per insertion (strong changes) and per kmer (proposals)
                self.strands_seen[pos, "+" + block].add(strand)
                for kmer in block:
                    self.strands_seen[pos, kmer].add(strand)
        # The position after the read's span counts too: an insertion at its far end lands there
        for pos in range(offset, offset + read.covered_len + 1):
            self.n_covering[pos] += 1

    def is_strong(self, pos: int, count: int, key: str) -> bool:
        """Whether `count` reads are over half of those covering `pos`, at least MIN_READ_SUPPORT, and on every strand."""
        majority = count >= MIN_READ_SUPPORT and count * 2 > self.n_covering[pos]
        return majority and _on_every_strand(self.strands_seen, pos, key, self.strands_present)

    def is_believable(self, pos: int, kmer: str) -> bool:
        """Whether `kmer` can be proposed at `pos`: seen there on every strand."""
        return _on_every_strand(self.strands_seen, pos, kmer, self.strands_present)


def _count_column_votes(consensus: str, alignments: list[AlignedRead]) -> _ColumnVotes:
    """Count what the reads have at each consensus position."""
    n = len(consensus)
    votes = _ColumnVotes(
        n_covering=[0] * (n + 1),
        observed=[Counter() for _ in range(n)],
        n_inserting=[0] * (n + 1),
        insertions=[Counter() for _ in range(n + 1)],
        strands_seen=defaultdict(set),
        strands_present={strand for *_, strand in alignments},
    )
    for read in alignments:
        votes.add_read(read)
    return votes


def _apply_strong_changes(consensus: str, votes: _ColumnVotes, length: int) -> str:
    """Apply every strong change at once and fit the result back to `length`; the consensus if it can't be."""
    segments, spare_insertions = _strong_change_segments(consensus, votes)
    return _fit_to_length(segments, spare_insertions, length) or consensus


def _strong_change_segments(consensus: str, votes: _ColumnVotes) -> tuple[list[tuple[str, int]], list[tuple[int, int, str]]]:
    """Return the new consensus as (kmers, support) segments, and the insertions that weren't strong as (support, index, kmers)."""
    segments: list[tuple[str, int]] = []
    spare_insertions: list[tuple[int, int, str]] = []
    # Insertion before each position, then the position; the extra round is the end
    for pos in range(len(consensus) + 1):
        if votes.n_inserting[pos]:
            block, count = votes.insertions[pos].most_common(1)[0]
            if votes.is_strong(pos, votes.n_inserting[pos], "+" + block):
                segments.append((block, count))
            else:
                spare_insertions.append((votes.n_inserting[pos], len(segments), block))
        if pos < len(consensus):
            # Keep the consensus kmer unless a strong majority has something else
            kmer, count = votes.observed[pos].most_common(1)[0] if votes.observed[pos] else (consensus[pos], 0)
            if kmer == consensus[pos] or not votes.is_strong(pos, count, kmer):
                segments.append((consensus[pos], votes.observed[pos][consensus[pos]]))
            elif kmer:  # A strong gap deletes the position
                segments.append((kmer, count))
    return segments, spare_insertions


def _fit_to_length(segments: list[tuple[str, int]], spare_insertions: list[tuple[int, int, str]], length: int) -> str | None:
    """Join the segments into `length` kmers: drop the least supported single kmers, or add the best supported spare insertions."""
    sequence = "".join(kmers for kmers, _ in segments)
    if len(sequence) > length:
        # Drop only single kmers, never part of an inserted block
        singles = sorted((i for i, (kmers, _) in enumerate(segments) if len(kmers) == 1), key=lambda i: segments[i][1])
        drop = set(singles[: len(sequence) - length])
        sequence = "".join(kmers for i, (kmers, _) in enumerate(segments) if i not in drop)
    while len(sequence) < length:
        if not spare_insertions:
            return None
        # Best supported spare insertion, cut to what's missing
        spare_insertions.sort()
        _, i, block = spare_insertions.pop()
        segments.insert(i, (block[: length - len(sequence)], 0))
        # Later indices shift by one
        spare_insertions = [(support, j + (j >= i), b) for support, j, b in spare_insertions]
        sequence = "".join(kmers for kmers, _ in segments)
    return sequence[:length]


def _apply_weak_changes(
    consensus: str,
    votes: _ColumnVotes,
    *,
    alignments: list[AlignedRead],
    read_distances: list[int],
    target_counts: dict[str, int],
    current_score: Score,
) -> str:
    """Apply the improving moves at disputed positions, each scored on its own; the consensus if there are none."""
    candidates = _candidate_moves(consensus, votes)
    moves = _find_improving_moves(
        consensus,
        candidates,
        alignments=alignments,
        read_distances=read_distances,
        target_counts=target_counts,
        current_score=current_score,
    )
    return _apply_spaced_moves(consensus, moves)


def _candidate_moves(consensus: str, votes: _ColumnVotes) -> set[Move]:
    """Return the moves to try at disputed positions: substitutions, and a swap either side.

    A position is disputed when any covering read differs from the consensus there: another kmer,
    a gap, or an insertion just before it. Substitutions try the other kmers the reads have there,
    and the kmers they insert just before it, if seen on every strand there. At high coverage
    (above ~40x), a substitution also needs at least MIN_SUPPORT_FRACTION of covering reads
    proposing the same change — low-support candidates never improve the objective.
    """
    n = len(consensus)
    candidates: set[Move] = set()
    for pos in range(n):
        disputed = votes.n_inserting[pos] or any(kmer != consensus[pos] for kmer in votes.observed[pos])
        if not disputed:
            continue
        # Other kmers here ("" is a gap) and kmers inserted just before
        proposed = {k for k in votes.observed[pos] if k and k != consensus[pos]} | {k for block in votes.insertions[pos] for k in block}
        # Filter only at high coverage; at low depth a single read can still improve the objective
        fraction_support = ceil(MIN_SUPPORT_FRACTION * votes.n_covering[pos])
        min_support = fraction_support if fraction_support > MIN_READ_SUPPORT else 0
        for k in proposed:
            if k == consensus[pos] or not votes.is_believable(pos, k):
                continue
            # Insertion-derived kmers have no observed count; keep them (rare, strand-checked)
            observed = votes.observed[pos].get(k, 0)
            if observed and observed < min_support:
                continue
            candidates.add(Move("sub", pos, k))
        # Swaps with either neighbour, if it differs
        if pos + 1 < n and consensus[pos] != consensus[pos + 1]:
            candidates.add(Move("swap", pos))
        if pos > 0 and consensus[pos - 1] != consensus[pos]:
            candidates.add(Move("swap", pos - 1))
    return candidates


def _find_improving_moves(
    consensus: str,
    candidates: set[Move],
    *,
    alignments: list[AlignedRead],
    read_distances: list[int],
    target_counts: dict[str, int],
    current_score: Score,
) -> list[ScoredMove]:
    """Return the candidates that lower the objective."""
    # Read position per consensus position, for cutting read windows
    consensus_to_read = [_consensus_to_read_positions(read) for read in alignments]
    # For the composition change
    counts = Counter(consensus)
    moves = []
    n_reads = len(alignments)
    for move in candidates:
        # Early-terminate: each remaining read improves the sum by at most 1 (OSA
        # distance), so if the partial delta exceeds the reads left, the move can't win
        new_distances = []
        delta = 0
        can_improve = True
        for i, (read, positions, distance) in enumerate(zip(alignments, consensus_to_read, read_distances, strict=True)):
            new_d = _distance_after_move(consensus, move, read, positions, distance)
            new_distances.append(new_d)
            delta += new_d - distance
            if delta > n_reads - i - 1:
                can_improve = False
                break
        if not can_improve:
            continue
        composition = current_score[1] + _composition_change(consensus, move, counts, target_counts)
        score = (sum(new_distances), composition, sum(d * d for d in new_distances))
        if score < current_score:
            moves.append((score, move))
    return moves


def _consensus_to_read_positions(read: AlignedRead) -> list[int]:
    """Return the read position aligned to each position of the covered part, plus the read's end."""
    positions = [len(read.kmers)] * (read.covered_len + 1)
    for op in read.opcodes:
        # Same length in the consensus and the read
        if op.tag in ("equal", "replace"):
            positions[op.src_start : op.src_end] = range(op.dest_start, op.dest_end)
        # Positions the read lacks map to where it continues
        elif op.tag == "delete":
            positions[op.src_start : op.src_end] = [op.dest_start] * (op.src_end - op.src_start)
    # A window from the consensus's start includes kmers the read inserts before it
    positions[0] = 0
    return positions


def _distance_after_move(consensus: str, move: Move, read: AlignedRead, consensus_to_read: list[int], distance: int) -> int:
    """Return a read's distance to the consensus after `move`, rescoring only the SCORE_WINDOW units around it."""
    # The move's span within the read's covered part (n units)
    start = move.pos - read.offset
    end = start + (1 if move.kind == "sub" else 2)
    n = len(consensus_to_read) - 1
    # Outside the read: distance unchanged
    if end <= 0 or start >= n:
        return distance
    # Straddles the read's end: rescore the whole read
    if start < 0 or end > n:
        return OSA.distance(_apply_move(consensus, move)[read.offset : read.offset + n], read.kmers)
    # A move only changes the alignment near it: rescore a window
    window_start, window_end = max(0, start - SCORE_WINDOW), min(n, end + SCORE_WINDOW)
    old_window = consensus[read.offset + window_start : read.offset + window_end]
    # Inline the move application: avoids NamedTuple creation and function call in the hot path
    idx = start - window_start
    if move.kind == "sub":
        new_window = old_window[:idx] + move.kmer + old_window[idx + 1 :]
    else:
        new_window = old_window[:idx] + old_window[idx + 1] + old_window[idx] + old_window[idx + 2 :]
    read_window = read.kmers[consensus_to_read[window_start] : consensus_to_read[window_end]]
    return distance - OSA.distance(old_window, read_window) + OSA.distance(new_window, read_window)


def _composition_change(consensus: str, move: Move, counts: Counter[str], target_counts: dict[str, int]) -> int:
    """Return the change in composition penalty from `move`; a swap doesn't change the composition."""
    if move.kind != "sub":
        return 0
    old, new = consensus[move.pos], move.kmer
    before = abs(counts[old] - target_counts.get(old, 0)) + abs(counts[new] - target_counts.get(new, 0))
    after = abs(counts[old] - 1 - target_counts.get(old, 0)) + abs(counts[new] + 1 - target_counts.get(new, 0))
    return after - before


def _apply_spaced_moves(consensus: str, moves: list[ScoredMove]) -> str:
    """Apply the moves best first (ties: leftmost), skipping any within two windows of one already applied."""
    applied: list[int] = []
    for _score, move in sorted(moves, key=lambda scored: (scored[0], scored[1].pos, scored[1])):
        # Windows don't overlap, so the scores still hold
        if all(abs(move.pos - applied_pos) >= 2 * SCORE_WINDOW + 2 for applied_pos in applied):
            applied.append(move.pos)
            consensus = _apply_move(consensus, move)
    return consensus


def _covered_part(consensus: str, read_len: int, read_type: str) -> tuple[str, int]:
    """Return the part of the consensus a read covers, and its offset into the consensus."""
    if read_type == "spanning":
        return consensus, 0
    if read_type == "left":
        return consensus[: min(read_len, len(consensus))], 0
    if read_type == "right":
        start = max(0, len(consensus) - read_len)
        return consensus[start:], start
    raise ValueError(read_type)


def _on_every_strand(strands_by_pos_kmer: dict[tuple[int, str], set[str]], pos: int, kmer: str, strands_present: set[str]) -> bool:
    """Whether every strand present shows `kmer` within STRAND_MARGIN of `pos` - see _initial_consensus and _ColumnVotes."""
    return {s for d in range(-STRAND_MARGIN, STRAND_MARGIN + 1) for s in strands_by_pos_kmer.get((pos + d, kmer), ())} >= strands_present


def _apply_move(consensus: str, move: Move) -> str:
    i = move.pos
    if move.kind == "sub":
        return consensus[:i] + move.kmer + consensus[i + 1 :]
    if move.kind == "swap":
        return consensus[:i] + consensus[i + 1] + consensus[i] + consensus[i + 2 :]
    raise ValueError(move)


def _target_composition(reads: list[EncodedRead]) -> dict[str, int]:
    """Per-kmer median count over the spanning reads.

    Zero-filled, so a kmer most reads lack gets median 0 and is dropped. Empty when nothing spans:
    a flank read's composition says nothing about the allele as a whole.
    """
    pool = [kmers for kmers, read_type, _ in reads if read_type == "spanning"]
    if not pool:
        return {}
    target_counts = {}
    for k in {c for kmers in pool for c in kmers}:
        # Median count (the upper one for an even number of reads)
        counts = sorted(kmers.count(k) for kmers in pool)
        target_counts[k] = counts[len(counts) // 2]
    return {k: c for k, c in target_counts.items() if c > 0}


def _composition_penalty(consensus: str, target_counts: dict[str, int]) -> int:
    if not target_counts:
        return 0
    counts = Counter(consensus)
    return sum(abs(counts[k] - target_counts.get(k, 0)) for k in set(target_counts) | set(counts))
