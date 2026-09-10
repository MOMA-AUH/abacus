# Greedy hill-climb search for a consensus repeat-unit sequence, used by consensus.py.
#
# Reads are kmer lists (one kmer per repeat unit) paired with a type: "spanning" covers the
# locus, "left"/"right" reach in from one flank and score against a trimmed seed. Each distinct
# kmer is encoded as one character, so the search runs on plain strings. Each iteration aligns
# every read to the seed once, proposes edits by vote, then rescores every candidate with a full
# distance recompute (`total_dist`'s `limit`/`suffix` bail out once a candidate can no longer beat
# the current best, so this stays cheap). A final pass fixes seed length, which the objective
# alone cannot pin down.
#
# Objective: summed OSA distance (Damerau-Levenshtein, restricted transpositions). The
# transposition discount is load-bearing - it makes a shifted interruption cost 1 rather than 2,
# which is what makes keeping an interruption beat flattening it. Plain Levenshtein loses that;
# unbounded Damerau scores the same but is ~40x slower.
#
# Partial kmers at read boundaries (a boundary kmer is only as long as the read actually covers)
# can look like real vocabulary; trim_partial_boundary_kmers drops one unless it's corroborated
# by an interior kmer elsewhere in the read set.

from collections import Counter, defaultdict
from itertools import accumulate, groupby

from rapidfuzz.distance import OSA, Opcodes
from rapidfuzz.distance import Levenshtein as RFLevenshtein

MARGIN = 1
MIN_READ_SUPPORT = 2
MIN_SLIPPAGE_RUN = 2

# One-char-per-kmer sequence, paired with "spanning" | "left" | "right", and strand "+" | "-"
EncodedRead = tuple[str, str, str]
# Raw kmer list, paired with "spanning" | "left" | "right", and strand "+" | "-"
KmerRead = tuple[list[str], str, str]
# (kind, position, ...) - see apply_op
Op = tuple

# Printable, non-reserved unicode range for kmer encoding. This encoding never reaches spoa
# (unlike equal_length_split.py's _USABLE_KMER_CHARS), so it isn't limited to single-byte-safe ASCII.
_USABLE_CHARS = [chr(i) for i in range(33, 0x110000) if chr(i).isprintable()][:20000]


def trim_seed_for(seed: str, read_len: int, read_type: str) -> tuple[str, int]:
    """Stretch of seed a read scores against, and its offset into the seed."""
    if read_type == "spanning":
        return seed, 0
    if read_type == "left":
        return seed[: min(read_len, len(seed))], 0
    if read_type == "right":
        start = max(0, len(seed) - read_len)
        return seed[start:], start
    raise ValueError(read_type)


def total_dist(
    seed: str,
    reads: list[EncodedRead],
    limit: int | None = None,
    suffix: list[int] | None = None,
) -> int:
    """Summed distance to every read, or `limit` once it provably reaches it. `suffix` - valid
    only for a seed ONE edit from the one it was built on - bounds what the unscored reads can
    still recover, since one edit moves any read's distance by at most 1.
    """
    total, n = 0, len(reads)
    for i, (kmers, rtype, _strand) in enumerate(reads):
        trimmed, _ = trim_seed_for(seed, len(kmers), rtype)
        total += OSA.distance(trimmed, kmers)
        if limit is not None and total + (suffix[i + 1] - (n - i - 1) if suffix else 0) >= limit:
            return limit
    return total


def median_length(reads: list[EncodedRead], spanning_only: bool) -> int | None:
    """Median read length; None if `spanning_only` and nothing spans. A flank read stops where
    its alignment ran out, so its length says nothing about the allele.
    """
    lens = sorted(len(k) for k, t, _ in reads if t == "spanning") or ([] if spanning_only else sorted(len(k) for k, _, _ in reads))
    return lens[len(lens) // 2] if lens else None


def strand_eligible_kmers(reads: list[EncodedRead]) -> set[str] | None:
    """Kmers seen on every strand present in `reads`, or None if only one strand is present.

    Guards against a strand-specific ONT error becoming the pooled majority. With one strand
    there's nothing to compare against, so nothing can be disqualified.
    """
    strands_present = {strand for _, _, strand in reads}
    if len(strands_present) <= 1:
        return None
    support: dict[str, set[str]] = defaultdict(set)
    for kmers, _rtype, strand in reads:
        for k in set(kmers):
            support[k].add(strand)
    return {k for k, strands in support.items() if strands >= strands_present}


def starting_seed(reads: list[EncodedRead]) -> str:
    """Medoid of the spanning reads, averaging distances per strand then summing so one strand's
    ONT errors can't dominate by sheer count. Falls back to most-observed kmer x median length,
    restricted to strand-eligible kmers, when nothing spans.
    """
    spanning_by_strand: dict[str, list[str]] = defaultdict(list)
    for kmers, rtype, strand in reads:
        if rtype == "spanning":
            spanning_by_strand[strand].append(kmers)
    if spanning_by_strand:
        groups = spanning_by_strand.values()
        return min(
            (s for group in groups for s in group),
            key=lambda s: sum(sum(OSA.distance(s, other) for other in group) / len(group) for group in groups),
        )

    counts = Counter(k for kmers, *_ in reads for k in kmers)
    eligible = strand_eligible_kmers(reads)
    if eligible:
        restricted = Counter({k: c for k, c in counts.items() if k in eligible})
        if restricted:
            counts = restricted
    length = median_length(reads, spanning_only=False)
    assert length is not None  # reads is non-empty here
    return counts.most_common(1)[0][0] * length


def apply_op(seed: str, op: Op) -> str:
    kind, i = op[0], op[1]
    if kind == "sub":
        return seed[:i] + op[2] + seed[i + 1 :]
    if kind == "del":
        return seed[:i] + seed[i + 1 :]
    if kind == "ins":
        return seed[:i] + op[2] + seed[i:]
    if kind == "swap":
        return seed[:i] + seed[i + 1] + seed[i] + seed[i + 2 :]
    raise ValueError(op)


def build_vocab(reads: list[EncodedRead], min_support: int = MIN_READ_SUPPORT) -> list[str]:
    counts = Counter(k for kmers, *_ in reads for k in set(kmers))
    return sorted(k for k, c in counts.items() if c >= min_support)


AlignedRead = tuple[str, int, Opcodes, str]


def align_reads(seed: str, reads: list[EncodedRead]) -> list[AlignedRead]:
    """One alignment per read -> (kmers, offset, opcodes, strand)."""
    aligned = []
    for kmers, rtype, strand in reads:
        trimmed, offset = trim_seed_for(seed, len(kmers), rtype)
        opcodes = RFLevenshtein.opcodes(trimmed, kmers)
        aligned.append((kmers, offset, opcodes, strand))
    return aligned


Votes = tuple[set[int], dict[tuple[int, str], int], dict[int, int], dict[tuple[int, str], int]]


def windowed_strands(strands_by_pos_kmer: dict[tuple[int, str], set[str]], pos: int, kmer: str) -> set[str]:
    """Union of strands voting for `kmer` within MARGIN of `pos` - see collect_votes."""
    return {s for d in range(-MARGIN, MARGIN + 1) for s in strands_by_pos_kmer.get((pos + d, kmer), ())}


def collect_votes(seed: str, aligned: list[AlignedRead], vocab_set: set[str]) -> Votes:
    """-> (positions, sub_votes[(pos, kmer)], del_votes[pos], ins_votes[(pos, kmer)]).

    Votes across reads, not each read's edit op as its own candidate: the alignment DP breaks ties
    arbitrarily, so no single read reliably names the fix a position needs, but reads agreeing on
    a real difference still outvote those that don't. `positions` feeds the swap scan.

    A sub/ins candidate also needs support from every strand present in `aligned`, checked within
    MARGIN of its own (position, kmer) rather than anywhere in the read set: a kmer can be a
    strand-specific error at one position and genuine at another, so eligibility must be checked
    per-position, not globally. The MARGIN window still tolerates ordinary registration slop
    (the same interruption landing a column apart across reads of different length).
    """
    sub_votes: dict[tuple[int, str], int] = defaultdict(int)
    del_votes: dict[int, int] = defaultdict(int)
    ins_votes: dict[tuple[int, str], int] = defaultdict(int)
    sub_strands: dict[tuple[int, str], set[str]] = defaultdict(set)
    ins_strands: dict[tuple[int, str], set[str]] = defaultdict(set)
    positions: set[int] = set()
    strands_present = {strand for *_, strand in aligned}
    for kmers, offset, opcodes, strand in aligned:
        for op in opcodes:
            if op.tag == "equal":
                continue
            positions.update(range(op.src_start + offset, max(op.src_end, op.src_start + 1) + offset))
            if op.tag == "replace":
                for d in range(op.src_end - op.src_start):
                    if kmers[op.dest_start + d] in vocab_set:
                        key = (op.src_start + d + offset, kmers[op.dest_start + d])
                        sub_votes[key] += 1
                        sub_strands[key].add(strand)
            elif op.tag == "delete":
                for d in range(op.src_end - op.src_start):
                    del_votes[op.src_start + d + offset] += 1
            else:
                for j in range(op.dest_start, op.dest_end):
                    if kmers[j] in vocab_set:
                        key = (op.src_start + offset, kmers[j])
                        ins_votes[key] += 1
                        ins_strands[key].add(strand)
    # drop candidates missing support from some strand within MARGIN of their position
    sub_votes = {key: v for key, v in sub_votes.items() if windowed_strands(sub_strands, *key) >= strands_present}
    ins_votes = {key: v for key, v in ins_votes.items() if windowed_strands(ins_strands, *key) >= strands_present}
    expanded = {p + d for p in positions for d in range(-MARGIN, MARGIN + 1) if 0 <= p + d <= len(seed)}
    return expanded | {len(seed)}, sub_votes, del_votes, ins_votes


Move = tuple[int, int, int, Op]


def find_improving_moves(
    seed: str,
    reads: list[EncodedRead],
    suffix: list[int],
    baseline: int,
    positions: set[int],
    sub_votes: dict[tuple[int, str], int],
    del_votes: dict[int, int],
    ins_votes: dict[tuple[int, str], int],
    min_support: int = MIN_READ_SUPPORT,
) -> list[Move]:
    """Supported candidates that lower the total, as (lo, hi, delta, op)."""
    n = len(seed)
    candidates: list[tuple[int, int, Op]] = [
        (pos, pos, ("sub", pos, k)) for (pos, k), votes in sub_votes.items() if votes >= min_support and pos < n and seed[pos] != k
    ]
    candidates += [(pos, pos + 1, ("swap", pos)) for pos in sorted(positions) if pos + 1 < n and seed[pos] != seed[pos + 1]]
    candidates += [(pos, pos, ("del", pos)) for pos, votes in del_votes.items() if votes >= min_support and pos < n]
    candidates += [(pos, pos, ("ins", pos, k)) for (pos, k), votes in ins_votes.items() if votes >= min_support]

    # a bailed-out score comes back as `baseline`, i.e. delta 0, i.e. rejected
    return [(lo, hi, delta, op) for lo, hi, op in candidates if (delta := baseline - total_dist(apply_op(seed, op), reads, baseline, suffix)) > 0]


def apply_batch(seed: str, accepted: list[Move]) -> str:
    result, i, n = "", 0, len(seed)
    for lo, hi, _delta, op in sorted(accepted, key=lambda m: m[0]):
        result += seed[i:lo]
        kind = op[0]
        if kind == "sub":
            result += op[2]
        elif kind == "ins":
            result += op[2] + (seed[lo] if lo < n else "")
        elif kind == "swap":
            result += seed[hi] + seed[lo]
        i = hi + 1 if kind != "ins" else lo + 1
    return result + seed[i:n]


def _try_moves(seed: str, reads: list[EncodedRead], moves: list[Move], baseline: int) -> tuple[str, int]:
    """Best-first batch of non-overlapping moves, dropping the weakest until it improves."""
    if not moves:
        return seed, baseline
    # best-first, skipping any move whose margin-expanded range touches an accepted one
    accepted: list[Move] = []
    ranges: list[tuple[int, int]] = []
    for lo, hi, delta, op in sorted(moves, key=lambda m: (-m[2], m[0])):
        if not any(r[0] <= hi + MARGIN and lo - MARGIN <= r[1] for r in ranges):
            accepted.append((lo, hi, delta, op))
            ranges.append((lo, hi))
    while accepted:
        new_seed = apply_batch(seed, accepted)
        new_total = total_dist(new_seed, reads, baseline)
        if new_total < baseline:
            return new_seed, new_total
        accepted.pop()  # drop the lowest-delta move (accepted is best-first) and retry
    return seed, baseline


def slippage_candidates(seed: str, grow: bool):
    """Seeds one unit longer (`grow`) or shorter, differing only in one run's length.

    Length errors come from slippage, which resizes a stretch of identical units rather than
    inventing one inside unrelated sequence - so one candidate per run, not per position. Growth
    needs MIN_SLIPPAGE_RUN, so a lone interruption never becomes a false repeat.
    """
    start = 0
    for _kmer, group in groupby(seed):
        length = len(list(group))
        if not grow:
            yield seed[:start] + seed[start + 1 :]
        elif length >= MIN_SLIPPAGE_RUN:
            yield seed[:start] + seed[start] + seed[start:]
        start += length


def repair_length(seed: str, reads: list[EncodedRead], target: int | None) -> str:
    """Close a pure length gap the hill-climb cannot see.

    Distance is flat in run length: against a read with e substitution errors, a seed k units
    short (k <= e) scores identically, since inserting the read's error unit costs what
    substituting it costs. So no alignment ever proposes the missing unit, and the objective
    cannot say when to stop either - hence `target`, without which a non-worsening step is always
    available and the seed grows past the flank reads that no longer constrain it.
    """
    if target is None:
        return seed
    baseline = total_dist(seed, reads)
    while len(seed) != target:
        for candidate in slippage_candidates(seed, grow=len(seed) < target):
            total = total_dist(candidate, reads, baseline + 1)
            if total <= baseline:
                baseline, seed = total, candidate
                break
        else:
            break  # no run resizes without making the fit worse
    return seed


def hill_climb_encoded(reads: list[EncodedRead], max_iters: int = 100, filter_vocab: bool = True) -> str:
    """Runs in single-char-per-kmer space; see hill_climb for the public, kmer-list API."""
    min_support = MIN_READ_SUPPORT if filter_vocab else 1
    vocab_set = set(build_vocab(reads, min_support=min_support))
    seed = starting_seed(reads)

    for _ in range(max_iters):
        # suffix[i] = summed distance over reads[i:], so suffix[0] is the seed's own total
        dists = [OSA.distance(trim_seed_for(seed, len(k), t)[0], k) for k, t, _ in reads]
        suffix = list(accumulate(reversed(dists), initial=0))[::-1]
        baseline = suffix[0]
        aligned = align_reads(seed, reads)
        votes = collect_votes(seed, aligned, vocab_set)
        moves = find_improving_moves(seed, reads, suffix, baseline, *votes, min_support)
        new_seed, new_total = _try_moves(seed, reads, moves, baseline)
        if new_total >= baseline and min_support > 1:
            # stalled at the noise-filtering threshold - a single supporting read still beats
            # nothing, so retry once accepting singleton-supported candidates before giving up.
            # Needed at low read depth: with too few reads to ever reach min_support, a
            # majority-vote starting seed would otherwise be unfixable by the read data it split from.
            moves = find_improving_moves(seed, reads, suffix, baseline, *votes, 1)
            new_seed, new_total = _try_moves(seed, reads, moves, baseline)
        if new_total >= baseline:
            break
        seed = new_seed

    return repair_length(seed, reads, median_length(reads, spanning_only=True))


def trim_partial_boundary_kmers(reads: list[KmerRead]) -> list[KmerRead]:
    """Drop a flanking read's boundary kmer - its last, for a left-flanking read; its first, for
    a right-flanking read - unless that exact kmer is also seen elsewhere as an interior kmer.

    A flanking read only reaches as far as the read itself goes, so its boundary kmer can be a
    genuine partial unit - shorter than a real motif copy, or otherwise unlike one - purely
    because the read ran out there, not because a short or unusual unit is real. Corroboration
    only counts from interior kmers, never from another read's own boundary, so two reads
    independently truncated at the same point can't validate each other.
    """
    interior_kmers = {k for kmers, rtype, _strand in reads for k in (kmers if rtype == "spanning" else kmers[:-1] if rtype == "left" else kmers[1:])}
    trimmed = []
    for kmers, rtype, strand in reads:
        if rtype == "left" and kmers and kmers[-1] not in interior_kmers:
            kmers = kmers[:-1]
        elif rtype == "right" and kmers and kmers[0] not in interior_kmers:
            kmers = kmers[1:]
        if kmers:
            trimmed.append((kmers, rtype, strand))
    return trimmed


def encode_reads(reads: list[KmerRead]) -> tuple[list[EncodedRead], dict[str, str]]:
    """kmer-list reads -> single-char-per-kmer reads, plus the decode map."""
    vocab_kmers = sorted({k for kmers, *_ in reads for k in kmers})
    if len(vocab_kmers) > len(_USABLE_CHARS):
        raise ValueError("too many distinct kmers for this quick encoding scheme")
    kmer_to_char = dict(zip(vocab_kmers, _USABLE_CHARS, strict=False))
    encoded = [("".join(kmer_to_char[k] for k in kmers), rtype, strand) for kmers, rtype, strand in reads]
    return encoded, {v: k for k, v in kmer_to_char.items()}


def hill_climb(reads: list[KmerRead], max_iters: int = 100, filter_vocab: bool = True) -> list[str]:
    """Consensus kmer list for `reads` (kmer list, "spanning"|"left"|"right" type, strand). Trims
    partial boundary kmers, encodes, runs the search, decodes.
    """
    encoded_reads, char_to_kmer = encode_reads(trim_partial_boundary_kmers(reads))
    result = hill_climb_encoded(encoded_reads, max_iters=max_iters, filter_vocab=filter_vocab)
    return [char_to_kmer[c] for c in result]
