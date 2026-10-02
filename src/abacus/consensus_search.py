# Fixed-length hill-climb search for a consensus repeat-unit sequence, used by consensus.py.
#
# Reads are kmer lists (one kmer per repeat unit) paired with a type: "spanning" covers the
# locus, "left"/"right" reach in from one flank and score against a trimmed seed. Each distinct
# kmer is encoded as one character, so the search runs on plain strings.
#
# The consensus length is supplied by the caller (haplotyping already estimates it), so the
# search never changes length: the seed is built at that length by weighted per-position vote,
# and the only moves are substitutions and adjacent swaps. Each iteration aligns every read to
# the seed once, marks positions where enough covering reads disagree, tries every kmer those
# reads propose there plus a swap either way, rescores every candidate against all reads
# (`score`'s `limit`/`suffix` bail out once a candidate can no longer beat the current best), and
# applies the single best.
#
# Objective, lexicographic: (summed OSA distance, composition penalty, summed squared distance).
# OSA's transposition discount makes a shifted interruption cost 1 rather than 2, which is what
# makes keeping an interruption competitive with flattening it - but it still ties them, since a
# substitution and a transposition both cost 1. Composition (distance from the reads' median
# kmer counts) breaks that tie against flattening, and the squared term breaks the remaining tie
# against "be one read exactly, be badly wrong about the rest".
#
# Partial kmers at read boundaries (a boundary kmer is only as long as the read actually covers)
# can look like real vocabulary; trim_partial_boundary_kmers drops one unless it's corroborated
# by an interior kmer elsewhere in the read set.

from collections import Counter, defaultdict
from itertools import accumulate
from math import exp, lgamma, log

from rapidfuzz.distance import OSA, Opcodes
from rapidfuzz.distance import Levenshtein as RFLevenshtein

MARGIN = 1
MIN_READ_SUPPORT = 2
DISAGREE_FRAC = 0.05

# One-char-per-kmer sequence, paired with "spanning" | "left" | "right", and strand "+" | "-"
EncodedRead = tuple[str, str, str]
# Raw kmer list, paired with "spanning" | "left" | "right", and strand "+" | "-"
KmerRead = tuple[list[str], str, str]
# ("sub", position, kmer) | ("swap", position) - swaps position with position + 1
Op = tuple
# (total OSA distance, composition penalty, summed squared distance)
Score = tuple[int, int, int]

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


def windowed_strands(strands_by_pos_kmer: dict[tuple[int, str], set[str]], pos: int, kmer: str) -> set[str]:
    """Union of strands proposing `kmer` within MARGIN of `pos` - see seed_by_vote and collect_disagreement."""
    return {s for d in range(-MARGIN, MARGIN + 1) for s in strands_by_pos_kmer.get((pos + d, kmer), ())}


def target_composition(reads: list[EncodedRead]) -> dict[str, int]:
    """Per-kmer median count over the spanning reads (zero-filled, so a kmer most reads lack
    gets median 0 and is dropped). Empty when nothing spans: a flank read's composition says
    nothing about the allele as a whole.
    """
    pool = [kmers for kmers, rtype, _ in reads if rtype == "spanning"]
    if not pool:
        return {}
    target = {}
    for k in {c for kmers in pool for c in kmers}:
        counts = sorted(kmers.count(k) for kmers in pool)
        target[k] = counts[len(counts) // 2]
    return {k: c for k, c in target.items() if c > 0}


def composition_penalty(seed: str, target: dict[str, int]) -> int:
    if not target:
        return 0
    counts = Counter(seed)
    return sum(abs(counts[k] - target.get(k, 0)) for k in set(target) | set(counts))


def score(
    seed: str,
    reads: list[EncodedRead],
    target: dict[str, int],
    limit: Score | None = None,
    suffix: list[int] | None = None,
) -> Score:
    """Lexicographic score against every read, or something worse than `limit` once the summed
    distance provably exceeds it. `suffix` - valid only for a seed ONE edit from the one it was
    built on - bounds what the unscored reads can still recover, since one edit moves any read's
    distance by at most 1. A candidate tying `limit` on distance is scored in full: the secondary
    keys may still prefer it.
    """
    total, squares, n = 0, 0, len(reads)
    for i, (kmers, rtype, _strand) in enumerate(reads):
        trimmed, _ = trim_seed_for(seed, len(kmers), rtype)
        d = OSA.distance(trimmed, kmers)
        total += d
        squares += d * d
        if limit is not None and total + (suffix[i + 1] - (n - i - 1) if suffix else 0) > limit[0]:
            return (limit[0] + 1, 0, 0)
    return (total, composition_penalty(seed, target), squares)


def binom_pmf(k: int, m: int, q: float) -> float:
    """P(m) for m ~ Binomial(k, q), computed in log-space.

    comb(k, m) alone overflows float for a read thousands of kmers off the target length, long
    before it's brought back down by the compensating q**m * (1 - q)**(k - m) factor - the product
    is a normal probability, but comb(k, m) computed first as an exact Python int is not. lgamma
    keeps the whole computation in log-space until the final exp, so a negligible-probability tail
    term safely underflows to 0.0 instead of overflowing on the way there.
    """
    if q <= 0.0:
        return 1.0 if m == 0 else 0.0
    if q >= 1.0:
        return 1.0 if m == k else 0.0
    log_pmf = lgamma(k + 1) - lgamma(m + 1) - lgamma(k - m + 1) + m * log(q) + (k - m) * log(1 - q)
    return exp(log_pmf) if log_pmf > -700 else 0.0


def seed_by_vote(reads: list[EncodedRead], length: int) -> str:
    """Per-position weighted vote at the target length.

    A flanking read is anchored at one end, so it names one kmer per position: left reads by
    offset from the left, right reads by offset from the right. A spanning read is anchored at
    both, and if it is k units off the target its k length errors sit somewhere along it - so
    seed position p maps to read position p + m (p - m for a short read), where m counts the
    errors before p. Assuming they are scattered uniformly, m ~ Binomial(k, p / (length - 1)):
    the read's single vote is spread over k + 1 slots, peaking at its left end for p near 0 and
    sliding to its right end for p near the end. Slots outside the read are dropped and the
    rest renormalised. Ties, and positions no read reaches, go to the most common kmer overall.

    A kmer can only win a position if it was voted for there (within MARGIN) by every strand
    present - the same guard collect_disagreement puts on proposals, so a strand-specific ONT
    error can't enter the seed by majority and then sit there because the objective likes it.
    """
    overall = Counter(c for kmers, _, _ in reads for c in kmers)
    strands_present = {strand for _, _, strand in reads}
    votes: list[dict[str, float]] = [defaultdict(float) for _ in range(length)]
    strands: dict[tuple[int, str], set[str]] = defaultdict(set)
    for kmers, rtype, strand in reads:
        n = len(kmers)
        for p in range(length):
            if rtype == "left":
                slots = [(kmers[p], 1.0)] if p < n else []
            elif rtype == "right":
                i = n - 1 - (length - 1 - p)
                slots = [(kmers[i], 1.0)] if 0 <= i < n else []
            else:
                k = abs(n - length)
                q = p / (length - 1) if length > 1 else 0.5
                slots = []
                for m in range(k + 1):
                    i = p + m if n >= length else p - m
                    if 0 <= i < n:
                        slots.append((kmers[i], binom_pmf(k, m, q)))
            norm = sum(w for _, w in slots)
            for kmer, w in slots:
                votes[p][kmer] += w / norm
                strands[p, kmer].add(strand)
    seed = []
    for p in range(length):
        eligible = {k: v for k, v in votes[p].items() if windowed_strands(strands, p, k) >= strands_present} or votes[p]
        if not eligible:
            seed.append(overall.most_common(1)[0][0])
            continue
        best = max(eligible.values())
        seed.append(max((k for k, v in eligible.items() if v == best), key=lambda k: overall[k]))
    return "".join(seed)


def apply_op(seed: str, op: Op) -> str:
    kind, i = op[0], op[1]
    if kind == "sub":
        return seed[:i] + op[2] + seed[i + 1 :]
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


def collect_disagreement(seed: str, aligned: list[AlignedRead], vocab_set: set[str]) -> tuple[set[int], dict[int, set[str]]]:
    """-> (active positions, proposed[pos] = kmers reads show there).

    A position is active when at least DISAGREE_FRAC of the reads covering it (never fewer than
    one) align a non-equal opcode onto it - replace and delete over the seed positions they
    consume, insert on the position it lands before. Every op counts the same: the alignment DP
    breaks ties arbitrarily, so which op a read used to describe a difference is not evidence
    of anything, only that it saw one.

    Proposed kmers come from replace and insert ops (an insertion before p often means p should
    be that kmer and the rest shifted). A proposal also needs support from every strand present,
    checked within MARGIN of its own (position, kmer) rather than anywhere in the read set: a
    kmer can be a strand-specific error at one position and genuine at another, so eligibility
    must be checked per-position, not globally.
    """
    covering: dict[int, int] = defaultdict(int)
    disagreeing: dict[int, int] = defaultdict(int)
    proposed: dict[int, set[str]] = defaultdict(set)
    strands: dict[tuple[int, str], set[str]] = defaultdict(set)
    strands_present = {strand for *_, strand in aligned}
    for kmers, offset, opcodes, strand in aligned:
        touched: set[int] = set()
        for op in opcodes:
            if op.tag == "equal":
                continue
            touched.update(range(op.src_start + offset, max(op.src_end, op.src_start + 1) + offset))
            if op.tag == "replace":
                for d in range(op.src_end - op.src_start):
                    key = (op.src_start + d + offset, kmers[op.dest_start + d])
                    proposed[key[0]].add(key[1])
                    strands[key].add(strand)
            elif op.tag == "insert":
                for j in range(op.dest_start, op.dest_end):
                    key = (op.src_start + offset, kmers[j])
                    proposed[key[0]].add(key[1])
                    strands[key].add(strand)
        for p in touched:
            disagreeing[p] += 1
        # opcodes[-1].src_end is the trimmed seed's length
        for p in range(offset, offset + (opcodes[-1].src_end if opcodes else 0)):
            covering[p] += 1
    active = {p for p, n in disagreeing.items() if n >= max(1, DISAGREE_FRAC * covering[p]) and p < len(seed)}
    proposed = {p: {k for k in ks if k in vocab_set and windowed_strands(strands, p, k) >= strands_present} for p, ks in proposed.items()}
    return active, proposed


Move = tuple[int, Score, Op]  # (position, score, op); position breaks score ties, earliest first


def find_improving_moves(
    seed: str,
    reads: list[EncodedRead],
    target: dict[str, int],
    suffix: list[int],
    baseline: Score,
    active: set[int],
    proposed: dict[int, set[str]],
) -> list[Move]:
    """Candidates that lower the score: every proposed sub at an active position, plus the swap
    either side of it.
    """
    n = len(seed)
    candidates: set[Op] = set()
    for pos in active:
        candidates.update(("sub", pos, k) for k in proposed.get(pos, ()) if seed[pos] != k)
        if pos + 1 < n and seed[pos] != seed[pos + 1]:
            candidates.add(("swap", pos))
        if pos > 0 and seed[pos - 1] != seed[pos]:
            candidates.add(("swap", pos - 1))
    # a bailed-out score comes back worse than baseline, i.e. rejected
    return [(op[1], sc, op) for op in candidates if (sc := score(apply_op(seed, op), reads, target, baseline, suffix)) < baseline]


def hill_climb_encoded(reads: list[EncodedRead], length: int, max_iters: int = 100) -> str:
    """Runs in single-char-per-kmer space; see hill_climb for the public, kmer-list API."""
    if length <= 0 or not reads:
        return ""
    vocab_set = set(build_vocab(reads))
    target = target_composition(reads)
    seed = seed_by_vote(reads, length)

    for _ in range(max_iters):
        # suffix[i] = summed distance over reads[i:], so suffix[0] is the seed's own total
        dists = [OSA.distance(trim_seed_for(seed, len(k), t)[0], k) for k, t, _ in reads]
        suffix = list(accumulate(reversed(dists), initial=0))[::-1]
        baseline = (suffix[0], composition_penalty(seed, target), sum(d * d for d in dists))
        active, proposed = collect_disagreement(seed, align_reads(seed, reads), vocab_set)
        moves = find_improving_moves(seed, reads, target, suffix, baseline, active, proposed)
        if not moves:
            break
        # one move per iteration: batching non-overlapping moves measured both slower and less accurate
        _pos, _score, op = min(moves, key=lambda m: (m[1], m[0]))
        seed = apply_op(seed, op)

    return seed


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


def hill_climb(reads: list[KmerRead], length: int, max_iters: int = 100, trim_boundaries: bool = True) -> list[str]:
    """Consensus kmer list of exactly `length` kmers for `reads` (kmer list,
    "spanning"|"left"|"right" type, strand). Trims partial boundary kmers, encodes, runs the
    search, decodes.

    `trim_boundaries=False` skips the boundary trim - for a segment that is always exactly one
    full kmer when a read reaches it at all (a locus break), "the boundary kmer might be a
    genuine partial copy" doesn't apply the way it does for an open-ended repeat run, and with a
    single read reaching it the trim would otherwise discard its only, complete kmer as
    uncorroborated.
    """
    if trim_boundaries:
        reads = trim_partial_boundary_kmers(reads)
    encoded_reads, char_to_kmer = encode_reads(reads)
    result = hill_climb_encoded(encoded_reads, length, max_iters=max_iters)
    return [char_to_kmer[c] for c in result]
