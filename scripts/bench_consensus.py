"""Benchmark abacus.consensus_search.find_consensus on simulated kmer reads with a known truth.

Usage (results are written to bench_results/ in the current directory):
    python scripts/bench_consensus.py run <suite> <label> [max_iters=N]   # suite: short | long | complex
    python scripts/bench_consensus.py compare <a> <b>                      # per-cell accuracy and time, b vs a

To compare a change: run the suites on the old code with one label, on the new code with another,
then compare the two labels.

Each cell: a true allele of L units; `depth` reads with per-unit errors (a one-base substitution,
a deleted unit, or a duplicated unit), motif boundaries placed one unit out of phase (the two
units at a boundary swapped), and on the minus strand a systematic error at a few fixed positions
(the strand gate should keep it out). With bad=0.2, a fifth of the reads are much noisier. The
search gets the true length, as the caller (haplotyping) supplies it. Accuracy = Levenshtein
distance between consensus and truth, in units, and as an error rate: that distance per unit of
the true allele.

Suites:
- short: CAG with random CAA interruptions, spanning reads only.
- long: the same alleles, long, with few reads, half of them flanking (alternately left and right,
  each covering 40-80% of the allele from its end).
- complex: RFC1-style alleles - runs of different 5-mer motifs (3-60 units each) with occasional
  single-unit interruptions - with half the reads flanking.
"""
import json
import random
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from statistics import mean

from rapidfuzz.distance import Levenshtein

import abacus.consensus_search as consensus_search

SUITES = {
    # allele kind, lengths, depths, bad fractions, reps, flanking
    "short": ("cag", [30, 100, 300], [5, 10, 30], [0.0, 0.2], 5, False),
    "long": ("cag", [500, 1500], [3, 6], [0.0], 5, True),
    "complex": ("rfc1", [200, 800], [5, 10], [0.0], 5, True),
}
RFC1_MOTIFS = ["AAAAG", "AAAGG", "AAGGG", "ACAGG", "AAGAG", "AGAGG"]
# (substitution, indel) rate per unit, and the rate at which a read places a motif boundary out of phase
ERRORS = {"low": (0.02, 0.01, 0.10), "high": (0.10, 0.05, 0.30)}
BAD_ERROR = (0.30, 0.10, 0.30)
RESULTS = Path.cwd() / "bench_results"


def mutate(kmer: str, rng: random.Random) -> str:
    i = rng.randrange(len(kmer))
    return kmer[:i] + rng.choice([b for b in "ACGT" if b != kmer[i]]) + kmer[i + 1 :]


def make_truth(kind: str, length: int, rng: random.Random) -> list[str]:
    if kind == "cag":
        return ["CAA" if rng.random() < 0.08 else "CAG" for _ in range(length)]
    # rfc1: runs of different motifs, with occasional single-unit interruptions
    truth: list[str] = []
    motif = rng.choice(RFC1_MOTIFS)
    while len(truth) < length:
        truth += [motif] * rng.randint(3, 60)
        if rng.random() < 0.3:
            truth.append(rng.choice([m for m in RFC1_MOTIFS if m != motif]))
        motif = rng.choice([m for m in RFC1_MOTIFS if m != motif])
    return truth[:length]


def shift_boundaries(truth: list[str], rate: float, rng: random.Random) -> list[str]:
    """Swap the two units at each motif boundary at `rate`, as a read placing it one unit out of phase."""
    units = list(truth)
    p = 0
    while p < len(units) - 1:
        if units[p] != units[p + 1] and rng.random() < rate:
            units[p], units[p + 1] = units[p + 1], units[p]
            p += 1  # don't move the same unit twice
        p += 1
    return units


def simulate_read(truth: list[str], sub: float, indel: float, swap: float, strand: str, systematic: dict[int, str], rng: random.Random) -> list[str]:
    read = []
    for p, kmer in enumerate(shift_boundaries(truth, swap, rng)):
        if strand == "-" and p in systematic and rng.random() < 0.5:
            read.append(systematic[p])
            continue
        r = rng.random()
        if r < indel / 2:
            continue
        if r < indel:
            read.extend([kmer, kmer])
        elif r < indel + sub:
            read.append(mutate(kmer, rng))
        else:
            read.append(kmer)
    return read


def run_cell(kind: str, length: int, depth: int, error: str, bad: float, rep: int, flanking: bool, settings: dict) -> dict:
    rng = random.Random(f"{kind}-{length}-{depth}-{error}-{bad}-{rep}")
    truth = make_truth(kind, length, rng)
    systematic = {p: mutate(truth[p], rng) for p in rng.sample(range(length), max(1, length // 50))}
    n_bad = round(depth * bad)
    reads = []
    for i in range(depth):
        sub, indel, swap = BAD_ERROR if i < n_bad else ERRORS[error]
        strand = "+" if i % 2 == 0 else "-"
        read_type = "spanning" if not flanking or i % 2 == 0 else ("left" if i % 4 == 1 else "right")
        cut = length if read_type == "spanning" else int(length * rng.uniform(0.4, 0.8))
        part, offset = (truth[:cut], 0) if read_type != "right" else (truth[-cut:], length - cut)
        part_systematic = {p - offset: kmer for p, kmer in systematic.items() if 0 <= p - offset < cut}
        reads.append((simulate_read(part, sub, indel, swap, strand, part_systematic, rng), read_type, strand))
    t0 = time.perf_counter()
    consensus = consensus_search.find_consensus(reads, length, **settings)
    elapsed = time.perf_counter() - t0
    return {"length": length, "depth": depth, "error": error, "bad": bad, "rep": rep, "dist": Levenshtein.distance(consensus, truth), "time": elapsed}


def error_rate(result: dict) -> float:
    """Distance to the truth per unit of the true allele."""
    return result["dist"] / result["length"]


def summary(results: list[dict]) -> str:
    return f"mean dist {mean(r['dist'] for r in results):.2f}, mean error rate {mean(map(error_rate, results)):.2%}"


def run(suite: str, label: str, settings: dict) -> None:
    kind, lengths, depths, bad_fracs, reps, flanking = SUITES[suite]
    cells = [(kind, *cell, flanking, settings) for cell in product(lengths, depths, ERRORS, bad_fracs, range(reps))]
    t0 = time.perf_counter()
    with Pool(8) as pool:
        results = pool.starmap(run_cell, cells)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{label}.json").write_text(json.dumps(results))
    print(f"{label}: {len(results)} runs in {time.perf_counter() - t0:.1f}s, {summary(results)}, total time {sum(r['time'] for r in results):.1f}s")


def compare(a: str, b: str) -> None:
    ra, rb = (json.loads((RESULTS / f"{x}.json").read_text()) for x in (a, b))
    key = lambda r: (r["length"], r["depth"], r["error"], r["bad"], r["rep"])
    rb_by_key = {key(r): r for r in rb}
    better = worse = 0
    cells: dict[tuple, list] = {}
    for r in ra:
        s = rb_by_key[key(r)]
        better += s["dist"] < r["dist"]
        worse += s["dist"] > r["dist"]
        cells.setdefault(key(r)[:4], []).append((r, s))
    print(f"{b} vs {a}: better {better}, worse {worse}, same {len(ra) - better - worse}")
    print(f"  {a}: {summary(ra)}")
    print(f"  {b}: {summary(rb)}")
    print(f"{'length':>6} {'depth':>5} {'error':>5} {'bad':>4} | {'dist':>6} {'err rate':>8} -> {'dist':>6} {'err rate':>8} | {'time':>7} -> {'time':>7}")
    for (length, depth, error, bad), pairs in cells.items():
        da, db = mean(p[0]["dist"] for p in pairs), mean(p[1]["dist"] for p in pairs)
        ea, eb = mean(error_rate(p[0]) for p in pairs), mean(error_rate(p[1]) for p in pairs)
        ta, tb = mean(p[0]["time"] for p in pairs), mean(p[1]["time"] for p in pairs)
        flag = "  <-- worse" if db > da else ""
        print(f"{length:>6} {depth:>5} {error:>5} {bad:>4} | {da:>6.2f} {ea:>8.2%} -> {db:>6.2f} {eb:>8.2%} | {ta:>6.3f}s -> {tb:>6.3f}s{flag}")
    print(f"total time: {sum(r['time'] for r in ra):.1f}s -> {sum(r['time'] for r in rb):.1f}s")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        settings = {k: int(v) for k, v in (arg.split("=") for arg in sys.argv[4:])}
        run(sys.argv[2], sys.argv[3], settings)
    else:
        compare(sys.argv[2], sys.argv[3])
