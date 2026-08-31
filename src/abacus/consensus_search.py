# Greedy hill-climb search for a consensus repeat-unit sequence, used by consensus.py.
#
# Reads are kmer lists (one kmer per repeat unit) paired with a type: "spanning" covers the
# locus, "left"/"right" reach in from one flank and score against a trimmed seed. Each distinct
# kmer is encoded as one character, so the search runs on plain strings. Each iteration aligns
# every read to the seed once and reuses that alignment twice: to propose edits by vote, and to
# keep scoring windows in register. A final pass fixes seed length, which the objective alone
# cannot pin down.
#
# Objective: summed OSA distance (Damerau-Levenshtein, restricted transpositions). The
# transposition discount is load-bearing - it makes a shifted interruption cost 1 rather than 2,
# which is what makes keeping an interruption beat flattening it. Plain Levenshtein loses that;
# unbounded Damerau scores the same but is ~40x slower.
#
# Partial kmers at read boundaries (a boundary kmer is only as long as the read actually covers)
# can look like real vocabulary; trim_partial_boundary_kmers drops one unless it's corroborated
# by an interior kmer elsewhere in the read set.

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from itertools import accumulate, groupby

from rapidfuzz.distance import OSA, Opcodes
from rapidfuzz.distance import Levenshtein as RFLevenshtein

MARGIN = 1
WINDOW_MARGIN = 15
MIN_READ_SUPPORT = 2
MIN_SLIPPAGE_RUN = 2

# One-char-per-kmer sequence, paired with "spanning" | "left" | "right"
EncodedRead = tuple[str, str]
# Raw kmer list, paired with "spanning" | "left" | "right"
KmerRead = tuple[list[str], str]
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
    for i, (kmers, rtype) in enumerate(reads):
        trimmed, _ = trim_seed_for(seed, len(kmers), rtype)
        total += OSA.distance(trimmed, kmers)
        if limit is not None and total + (suffix[i + 1] - (n - i - 1) if suffix else 0) >= limit:
            return limit
    return total


def median_length(reads: list[EncodedRead], spanning_only: bool) -> int | None:
    """Median read length; None if `spanning_only` and nothing spans. A flank read stops where
    its alignment ran out, so its length says nothing about the allele.
    """
    lens = sorted(len(k) for k, t in reads if t == "spanning") or ([] if spanning_only else sorted(len(k) for k, _ in reads))
    return lens[len(lens) // 2] if lens else None


def starting_seed(reads: list[EncodedRead]) -> str:
    """Most-observed kmer x median read length: neutral, unlike a real read and its own errors.

    Counts spanning reads only, when any exist. A spanning read sees the whole locus; a flanking
    read only sees its own side, so letting enough of them outvote a single spanning read would
    start the search on a motif no read looking at the whole picture actually supports.
    """
    spanning = [(kmers, t) for kmers, t in reads if t == "spanning"]
    counts = Counter(k for kmers, _ in (spanning or reads) for k in kmers)
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
    counts = Counter(k for kmers, _ in reads for k in set(kmers))
    return sorted(k for k, c in counts.items() if c >= min_support)


AlignedRead = tuple[str, int, list[int], list[int], Opcodes]


def align_reads(seed: str, reads: list[EncodedRead]) -> list[AlignedRead]:
    """One alignment per read -> (kmers, offset, pos_map, anchors, opcodes).

    pos_map[i] is the read index trimmed-seed index i aligns to; raw seed coordinates drift out of
    register as soon as a read carries an indel. `anchors` mark `equal` blocks, where the two
    provably line up, so a window cut there splits the alignment cleanly.
    """
    aligned = []
    for kmers, rtype in reads:
        trimmed, offset = trim_seed_for(seed, len(kmers), rtype)
        opcodes = RFLevenshtein.opcodes(trimmed, kmers)
        pos_map = [0] * (len(trimmed) + 1)
        anchors = [0]
        for op in opcodes:
            if op.tag == "insert":
                continue
            for d in range(op.src_end - op.src_start):
                pos_map[op.src_start + d] = op.dest_start if op.tag == "delete" else op.dest_start + d
            if op.tag == "equal":
                anchors.extend(range(op.src_start, op.src_end))
        pos_map[len(trimmed)] = len(kmers)
        anchors.append(len(trimmed))
        aligned.append((kmers, offset, pos_map, anchors, opcodes))
    return aligned


Votes = tuple[set[int], dict[tuple[int, str], int], dict[int, int], dict[tuple[int, str], int]]


def collect_votes(seed: str, aligned: list[AlignedRead], vocab_set: set[str]) -> Votes:
    """-> (positions, sub_votes[(pos, kmer)], del_votes[pos], ins_votes[(pos, kmer)]).

    Votes across reads, not each read's edit op as its own candidate: the alignment DP breaks ties
    arbitrarily, so no single read reliably names the fix a position needs, but reads agreeing on
    a real difference still outvote those that don't. `positions` feeds the swap scan.
    """
    sub_votes: dict[tuple[int, str], int] = defaultdict(int)
    del_votes: dict[int, int] = defaultdict(int)
    ins_votes: dict[tuple[int, str], int] = defaultdict(int)
    positions: set[int] = set()
    for kmers, offset, _pos_map, _anchors, opcodes in aligned:
        for op in opcodes:
            if op.tag == "equal":
                continue
            positions.update(range(op.src_start + offset, max(op.src_end, op.src_start + 1) + offset))
            if op.tag == "replace":
                for d in range(op.src_end - op.src_start):
                    if kmers[op.dest_start + d] in vocab_set:
                        sub_votes[(op.src_start + d + offset, kmers[op.dest_start + d])] += 1
            elif op.tag == "delete":
                for d in range(op.src_end - op.src_start):
                    del_votes[op.src_start + d + offset] += 1
            else:
                for j in range(op.dest_start, op.dest_end):
                    if kmers[j] in vocab_set:
                        ins_votes[(op.src_start + offset, kmers[j])] += 1
    expanded = {p + d for p in positions for d in range(-MARGIN, MARGIN + 1) if 0 <= p + d <= len(seed)}
    return expanded | {len(seed)}, sub_votes, del_votes, ins_votes


def windowed_delta(seed: str, op: Op, lo: int, hi: int, aligned: list[AlignedRead]) -> int:
    """Improvement from `op`, scored on a local window. Length-preserving ops ONLY: an indel
    shifts registration for everything downstream, which no fixed window sees (measured: a
    systematic ~1 per read), so those get scored exactly instead.
    """
    n = len(seed)
    w_lo, w_hi = max(0, lo - WINDOW_MARGIN), min(n, hi + 1 + WINDOW_MARGIN)
    candidate = apply_op(seed, op)

    total = 0
    for kmers, offset, pos_map, anchors, _opcodes in aligned:
        trimmed_len = len(pos_map) - 1
        a = min(max(w_lo - offset, 0), trimmed_len)
        b = min(max(w_hi - offset, 0), trimmed_len)
        if a >= b:
            continue
        a = anchors[max(bisect_right(anchors, a) - 1, 0)]
        b = anchors[min(bisect_left(anchors, b), len(anchors) - 1)]
        seed_window = seed[offset + a : offset + b]
        cand_window = candidate[offset + a : offset + b]
        if seed_window == cand_window:
            continue
        r_window = kmers[pos_map[a] : pos_map[b]]
        total += OSA.distance(seed_window, r_window) - OSA.distance(cand_window, r_window)
    return total


Move = tuple[int, int, int, Op]


def find_improving_moves(
    seed: str,
    reads: list[EncodedRead],
    aligned: list[AlignedRead],
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
    windowed: list[tuple[int, int, Op]] = [
        (pos, pos, ("sub", pos, k)) for (pos, k), votes in sub_votes.items() if votes >= min_support and pos < n and seed[pos] != k
    ]
    windowed += [(pos, pos + 1, ("swap", pos)) for pos in sorted(positions) if pos + 1 < n and seed[pos] != seed[pos + 1]]
    exact: list[tuple[int, int, Op]] = [(pos, pos, ("del", pos)) for pos, votes in del_votes.items() if votes >= min_support and pos < n]
    exact += [(pos, pos, ("ins", pos, k)) for (pos, k), votes in ins_votes.items() if votes >= min_support]

    moves = [(lo, hi, delta, op) for lo, hi, op in windowed if (delta := windowed_delta(seed, op, lo, hi, aligned)) > 0]
    # a bailed-out exact score comes back as `baseline`, i.e. delta 0, i.e. rejected
    moves += [(lo, hi, delta, op) for lo, hi, op in exact if (delta := baseline - total_dist(apply_op(seed, op), reads, baseline, suffix)) > 0]
    return moves


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
    """Apply the batch, falling back to the single best move; unchanged if neither helps."""
    if not moves:
        return seed, baseline
    # best-first, skipping any move whose margin-expanded range touches an accepted one
    accepted: list[Move] = []
    ranges: list[tuple[int, int]] = []
    for lo, hi, delta, op in sorted(moves, key=lambda m: (-m[2], m[0])):
        if not any(r[0] <= hi + MARGIN and lo - MARGIN <= r[1] for r in ranges):
            accepted.append((lo, hi, delta, op))
            ranges.append((lo, hi))
    new_seed = apply_batch(seed, accepted)
    new_total = total_dist(new_seed, reads, baseline)
    if new_total >= baseline:
        new_seed = apply_op(seed, max(moves, key=lambda m: m[2])[3])
        new_total = total_dist(new_seed, reads, baseline)
    return (seed, baseline) if new_total >= baseline else (new_seed, new_total)


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
        dists = [OSA.distance(trim_seed_for(seed, len(k), t)[0], k) for k, t in reads]
        suffix = list(accumulate(reversed(dists), initial=0))[::-1]
        baseline = suffix[0]
        aligned = align_reads(seed, reads)
        votes = collect_votes(seed, aligned, vocab_set)
        moves = find_improving_moves(seed, reads, aligned, suffix, baseline, *votes, min_support)
        new_seed, new_total = _try_moves(seed, reads, moves, baseline)
        if new_total >= baseline and min_support > 1:
            # stalled at the noise-filtering threshold - a single supporting read still beats
            # nothing, so retry once accepting singleton-supported candidates before giving up.
            # Needed at low read depth: with too few reads to ever reach min_support, a
            # majority-vote starting seed would otherwise be unfixable by the read data it split from.
            moves = find_improving_moves(seed, reads, aligned, suffix, baseline, *votes, 1)
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
    interior_kmers = {k for kmers, rtype in reads for k in (kmers if rtype == "spanning" else kmers[:-1] if rtype == "left" else kmers[1:])}
    trimmed = []
    for kmers, rtype in reads:
        if rtype == "left" and kmers and kmers[-1] not in interior_kmers:
            kmers = kmers[:-1]
        elif rtype == "right" and kmers and kmers[0] not in interior_kmers:
            kmers = kmers[1:]
        if kmers:
            trimmed.append((kmers, rtype))
    return trimmed


def encode_reads(reads: list[KmerRead]) -> tuple[list[EncodedRead], dict[str, str]]:
    """kmer-list reads -> single-char-per-kmer reads, plus the decode map."""
    vocab_kmers = sorted({k for kmers, _ in reads for k in kmers})
    if len(vocab_kmers) > len(_USABLE_CHARS):
        raise ValueError("too many distinct kmers for this quick encoding scheme")
    kmer_to_char = dict(zip(vocab_kmers, _USABLE_CHARS, strict=False))
    encoded = [("".join(kmer_to_char[k] for k in kmers), rtype) for kmers, rtype in reads]
    return encoded, {v: k for k, v in kmer_to_char.items()}


def hill_climb(reads: list[KmerRead], max_iters: int = 100, filter_vocab: bool = True) -> list[str]:
    """Consensus kmer list for `reads` (kmer list, "spanning"|"left"|"right" type). Trims partial
    boundary kmers, encodes, runs the search, decodes.
    """
    encoded_reads, char_to_kmer = encode_reads(trim_partial_boundary_kmers(reads))
    result = hill_climb_encoded(encoded_reads, max_iters=max_iters, filter_vocab=filter_vocab)
    return [char_to_kmer[c] for c in result]
