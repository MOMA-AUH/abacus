# Abacus

[![CI](https://github.com/MOMA-AUH/abacus/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/MOMA-AUH/abacus/actions/workflows/ci.yml)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/abacus-str.svg)](https://anaconda.org/bioconda/abacus-str)
[![Bioconda downloads](https://img.shields.io/conda/dn/bioconda/abacus-str.svg?label=downloads)](https://anaconda.org/bioconda/abacus-str)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

<img src="./img/logo.png" width="150">

## Description

Abacus analyzes STR (Short Tandem Repeat) data from Long-Read Sequencing. It targets Oxford Nanopore (ONT) data and has also been tested on Pacific Biosciences (PacBio) data. The main goal of Abacus is to provide a user-friendly interface for analyzing STR data and to provide a comprehensive report of the analysis results.

Abacus converts each entry of an STR catalog (JSON) into a graph, then re-maps relevant reads from an aligned BAM/CRAM file onto that graph using [minigraph](https://github.com/lh3/minigraph). Haplotyping uses the resulting repeat counts and sequences to infer allele groups, refine their consensus sequences, and call STR alleles.

### Haplotyping algorithm

Each haplotype requires `--min-reads-per-haplotype` usable reads (default: `2`). Diploid loci with fewer than twice this minimum are fitted as one homozygous (HOM) group; fewer than the minimum produce no call. At the default, two or three reads support HOM; four can support two haplotypes. Chromosome and sex determine biological ploidy.

1. **Allele grouping and filtering:** a Gaussian mixture model groups reads by repeat count. Groups and estimates remain provisional until consensus refinement finishes. Unsupported groups and length outliers are removed, then groups are refitted. Length outliers fall outside robust bounds and the wider window: `median × (1 ± tolerance)` (`--length-outlier-tolerance`) or `median ± bases` (`--length-outlier-tolerance-bases`). Filtering requires more than `--min-n-length-outlier-detection` spanning reads per group and removes at most one spanning and one overly long flanking read per iteration.

2. **Heterozygosity test:**

   - **a. Length test:** a likelihood ratio test compares homozygous and heterozygous models (one Gaussian versus two). Significant results (p < `--heterozygosity-alpha`) retain H1/H2; otherwise reads become HOM and the sequence test is attempted.

   - **b. Sequence test:** with at least `--min-haplotyping-depth` spanning reads (default: `10`), a partial-order alignment (POA) identifies the most variable kmer position. A binomial test checks the two commonest kmers for a 50/50 ratio. If p ≥ `--equal-length-alpha` and both groups meet minimum support, reads are split into H1/H2; reads carrying neither kmer are filtered (`not_split_base`). Otherwise all reads remain HOM.

3. **Consensus building:** each haplotype's consensus is built by hill-climbing over read kmers, independently for each satellite and break, then concatenating the results. Satellite lengths match the fitted repeat counts.

4. **Consensus refinement:** spanning and flanking reads are reassigned to a strictly closer consensus, preserving ties. Spanning-read moves trigger parameter and length refitting. Minimum support is rechecked after splitting and reassignment; unsupported groups are removed before rebuilding consensus.

### Reporting results

Results are saved as an HTML report (STR loci, called alleles, and visualizations) and a VCF file with the haplotyping calls (see [VCF output](#vcf-output)). `--add-consensus-to-vcf` adds the consensus sequences to the VCF output. For large catalogs, report generation can be skipped by omitting the `--report` argument.

## Installation

### From conda
```sh
conda install bioconda::abacus-str
```

### From source

Use `env.yml` to install all dependencies; `pip` alone does not install external tools.

```sh
git clone https://github.com/MOMA-AUH/abacus
cd abacus
conda env create -f env.yml -n abacus
conda activate abacus
python -m pip install --no-deps .
```

For development, clone with `--branch develop` and add `-e` to the pip command for an editable installation.

To update, activate the environment, run `git pull --ff-only` and `conda env update -f env.yml -n abacus`, then repeat the pip command.

## Basic usage
Core arguments:

- `--bam`: BAM or CRAM file with aligned reads.
- `--ref`: reference FASTA used to align the reads.
- `--report` *(optional)*: HTML output path. Omit to skip report generation (useful for large catalogs).
- `--vcf`: VCF output path for the STR haplotyping results.
- `--sample-id`: sample identifier.
- `--sex`: sample sex, `XX` or `XY` (default: `XX`).
- `--loci-subset`: restrict to a subset of loci. Repeat for multiple loci.
- `--str-catalog` *(optional)*: path to a custom STR catalog JSON. Defaults to the built-in catalog. Run `abacus --show-catalog` to print the built-in catalog to stdout as a starting point.

### Example 1: Analyze all loci
```sh
abacus \
    --bam input.bam \
    --ref reference.fa \
    --report output.html \
    --vcf output.vcf \
    --sample-id my_sample
```

### Example 2: Analyze a subset of loci (FGF14 and RFC1) in male
```sh
abacus \
    --bam input.bam \
    --ref reference.fa \
    --report output.html \
    --vcf output.vcf \
    --sample-id my_sample \
    --sex XY \
    --loci-subset FGF14 \
    --loci-subset RFC1
```

### Example 3: Use a custom catalog
```sh
# Save the built-in catalog to use as a starting point
abacus --show-catalog > my_catalog.json

# Run with a custom catalog
abacus \
    --bam input.bam \
    --ref reference.fa \
    --str-catalog my_catalog.json \
    --report output.html \
    --vcf output.vcf \
    --sample-id my_sample
```

### Configuration parameters
Fine-tuning knobs for the analysis:

#### Graph and Anchor Parameters
- `--anchor-length`: length of the left/right anchor sequences. Default: `500`.
- `--min-anchor-overlap`: minimum read/anchor overlap. Default: `200`.
- `--min-end-qual`: minimum base quality at read ends, used for trimming. Default: `17`.
- `--trim-window-size`: window size for trimming low-quality bases. Default: `10`.
- `--max-trim`: max bases trimmed from read ends. Default: `50`.

#### Quality Control Parameters
- `--min-str-qual`: minimum mean base quality in the STR region; lower is filtered out. Default: `17`.
- `--tol-str-qual`: tolerance for mean base quality in the STR region. Default: `35`.
- `--min-q10-str-quality`: minimum Q10 base quality in the STR region. Default: `7`.
- `--tol-q10-str-quality`: tolerance for Q10 base quality in the STR region. Default: `25`.
- `--max-error-rate`: max allowed error rate in the STR region; higher is filtered out. Default: `0.01`.
- `--tol-error-rate`: tolerance for error rate in the STR region. Default: `0.005`.
- `--tol-ref-divergence`: tolerance for reference divergence in the STR region; reads are only removed as divergence outliers above it. Default: `0.20`.
- `--length-outlier-tolerance`: fraction of the haplotype median STR base-pair length; reads within `median × (1 ± tolerance)` are always kept, even outside the Tukey-fence bounds. Default: `0.10` (10%).
- `--min-n-qc-filtering`: minimum read calls required to run statistical QC filtering. Default: `10`.
- `--min-n-length-outlier-detection`: minimum spanning reads per haplotype group to run length outlier detection. Default: `5`.

#### Haplotype Parameters
- `--min-reads-per-haplotype`: minimum usable reads required per reported haplotype; groups below this support are removed. Must be at least `1`. Default: `2`.
- `--min-haplotyping-depth`: minimum spanning-read depth to attempt the equal-length sequence split. Separate from `--min-reads-per-haplotype`. Default: `10`.
- `--heterozygosity-alpha`: sensitivity cutoff for the length-based heterozygosity test. Default: `0.05`.
- `--equal-length-alpha`: sensitivity cutoff for the equal-length sequence split test (backup test for equal-length haplotypes). Default: `0.05`.

#### Coverage Parameters
- `--downsample`: randomly downsample reads per locus above this count; useful for high-coverage data (e.g. PacBio PureTarget). `0` disables downsampling. Default: `1000`.
- `--downsample-seed`: random seed for downsampling, for reproducibility. Default: `42`.

#### Output Options
- `--log-file`: path to the log file. Default: `abacus.log`.
- `--keep-summaries`: keep consensus, haplotype, heterozygosity test, and final/test parameter summary CSVs. Default: `False`.
- `--keep-read-summaries`: keep the read-level tables `reads.csv` and `filtered_reads.csv`. Default: `False`.
- `--add-consensus-to-vcf`: add the consensus sequence of each allele (`CONS`) to the VCF output. Default: `False`.

Retained tables are saved in `tmp_abacus_<vcf-stem>/` beside the VCF. Combine both flags to keep all tables. Reports generate all tables temporarily; otherwise, only requested tables are generated. Intermediate files are preserved on failure.

Outputs use buffered writes, flushed every 100 completed loci and on closure. Forced termination may lose buffered results.

## VCF output

The VCF follows [VCFv4.5](https://samtools.github.io/hts-specs/VCFv4.5.pdf) and uses its `<CNV:TR>` tandem repeat representation. There is one record per satellite (repeat) of a locus, and one `<CNV:TR>` ALT allele per haplotype: `H1,H2` for a heterozygous locus, a single allele for a homozygous one. Every allele is reported as ALT, also when it matches the reference, so the repeat count estimate is always present.

Example: the HTT CAG satellite, where H1 is `(CAG)9CAA(CAG)8` and H2 is `(CAG)42`:

```
chr4  3074876  Abacus.HTT.1  C  <CNV:TR>,<CNV:TR>  .  PASS  SVLEN=57,57;RN=3,1;RUS=CAG,CAA,CAG,CAG;RUC=9,1,8,42;LPSRUS=CAG,CAG;LPSRUC=9,42;TRUC=18.1,41.7;CITRUC=-0.9,1.1,-2.4,3;REFDIV=0.023,0.006;LOCUSID=HTT;REPID=HTT_CAG  GT:PS:DP:AD:ADSP:ADFL  1|2:3074876:30:0,18,12:0,15,9:0,3,3
```

| Column / field | Description |
|---|---|
| `POS`, `REF` | The padding base immediately preceding the repeat. |
| `ID` | `Abacus.<LocusId>.<satellite index>`. |
| `INFO/SVLEN` | Reference length of the repeat (`1` for a repeat absent from the reference). |
| `INFO/RN` | Number of repeat sequences in each allele. `0` for an allele without repeat sequences in this satellite. |
| `INFO/RUS`, `INFO/RUC` | Repeat unit sequence and count of each repeat sequence, flattened in allele order. The consensus is split into runs of identical repeat units, so interruptions (e.g. `CAA`) and motif changes (e.g. RFC1 `AAAAG` to `AAGGG`) get their own entry. |
| `INFO/LPSRUS`, `INFO/LPSRUC` | Repeat unit sequence and count of the longest pure segment of each allele (see [Longest pure segment](#longest-pure-segment)). `LPSRUC` is `0` for an allele without repeat sequences in this satellite; `LPSRUS` stays `.` — no segment, so no repeat unit. |
| `INFO/TRUC` | Estimated total number of repeat units in each allele. |
| `INFO/CITRUC` | Confidence interval around `TRUC`, as lower and upper offsets from it for each allele. |
| `INFO/REFDIV` | Divergence (0–1) of each allele's consensus from the catalog locus structure. Informational: a high value means the allele doesn't look like the catalog motif (e.g. a different or interrupted repeat), so its repeat counts should be read with care. |
| `INFO/CONS` | Consensus sequence of each allele. Only with `--add-consensus-to-vcf`. |
| `INFO/LOCUSID`, `INFO/REPID` | The catalog `LocusId` and `VariantId`. |
| `INFO/STRSTATUS` | `normal`, `intermediate` or `pathogenic` for each allele, from `TRUC` and the catalog thresholds (see [Thresholds](#thresholds)). Only for satellites with thresholds. |
| `INFO/BENIGNMAX`, `INFO/PATHOGENICMIN` | The catalog `BenignMax` and `PathogenicMin`. Only when set for the satellite. |
| `FORMAT/GT` | `1\|2` (phased) for a heterozygous locus, `1/1` for a homozygous one. |
| `FORMAT/PS` | Phase set, shared by all satellite records of a heterozygous locus, so H1/H2 are consistent across them. |
| `FORMAT/DP` | Number of reads assigned to the alleles. |
| `FORMAT/AD` | Reads per allele, starting with the reference allele, which is always `0`. |
| `FORMAT/ADSP`, `FORMAT/ADFL` | Spanning and flanking reads per allele, starting with the reference allele like `AD`. |

### No-calls

A locus with no reads, or with all reads filtered out, gets a `./.` record with `FILTER` `NoReads` or `NoCall`. `DP` is the number of reads at the locus; the log lists why they were removed.

### Longest pure segment

`LPSRUS`/`LPSRUC` give the longest pure segment (LPS) of each allele, i.e. the longest entry in `RUS`/`RUC`. The idea is from [Danzi et al. (2025)](https://doi.org/10.1101/2025.01.06.631535), who found the LPS to be the best genome-wide predictor of pathogenicity. As in the paper, the LPS can be used to rank loci by their variability in a population (PLVI), and to prioritise alleles with an LPS that is long compared to the population. Population LPS data per locus can be found in [TRExplorer](https://trexplorer.broadinstitute.org).

## The STR catalog

The STR catalog is a JSON file describing the STR loci to analyze. Each entry contains:

- `LocusId`: unique identifier for the locus, used to refer to it in the analysis results.
- `LocusStructure`: the locus structure, with each repeat unit in parentheses followed by `*`. E.g. ATXN1: `(CTG)*`. Repeat units can be any length and use [IUPAC](https://en.wikipedia.org/wiki/International_Union_of_Pure_and_Applied_Chemistry) base symbols (e.g. `N`, `Y`), and the structure can include non-repeating flanking/interrupting sequence — e.g. HTT: `(CAG)*CAACAG(CCG)*`, where `CAACAG` interrupts the two repeat units. The `|` operator specifies alternative repeat sequences within a unit, e.g. `(CAG|CAA)*` counts both `CAG` and `CAA` toward the repeat count.
- `ReferenceRegion`: the locus's region(s) in the reference genome, as `chr:start-end` (0-based, half-open, like BED). A list of regions can be given to cover the full structure.

A built-in catalog ships with the package and is used by default. Run `abacus --show-catalog > my_catalog.json` to export it as a starting point, or write your own from scratch following the format above. Example:

```json
[
    {
        "LocusId": "ATXN1",
        "LocusStructure": "(CTG)*",
        "ReferenceRegion": "chr6:16327635-16327722"
    },
    {
        "LocusId": "HTT",
        "LocusStructure": "(CAG)*CAACAG(CCG)*",
        "ReferenceRegion": [
            "chr4:3074876-3074933",
            "chr4:3074939-3074966"
        ]
    }
]
```

Entries with a `_complex` suffix (e.g. `FGF14_complex`, `RFC1_complex`) are more detailed versions of the plain entry with the same name, declaring known alternative motifs, interruptions and flanking sequences. See [Notes on FGF14](#notes-on-fgf14) and [Notes on RFC1](#notes-on-rfc1).

When declaring alternative motifs with `|`, avoid including a motif together with one of its rotations (e.g. `ACAGG` and `GGACA`, the same repeat read from a different starting base). The aligner then picks between them arbitrarily, so the same allele can be reported as different motifs.

### Thresholds

Optionally, a locus can have repeat-count thresholds:

- `BenignMax`: largest benign repeat unit count.
- `PathogenicMin`: smallest pathogenic repeat unit count.

Each count estimate is then classified as `normal` (≤ `BenignMax`), `intermediate` (between the two) or `pathogenic` (≥ `PathogenicMin`). Either field can be left out: without `BenignMax` there is no intermediate range, and without `PathogenicMin` nothing is classified as pathogenic. In the report summary table, intermediate count estimates are highlighted in orange and pathogenic ones in red, and the VCF gets the `STRSTATUS`, `BENIGNMAX` and `PATHOGENICMIN` fields (see [VCF output](#vcf-output)).

For a single-satellite locus, give each threshold as a number. For a locus with several satellites, give a list with one value per satellite, in `LocusStructure` order, using `null` for satellites without a threshold. E.g. HTT, with thresholds on the CAG repeat only (values from [STRchive](https://strchive.org/loci/HD_HTT)):

```json
[
    {
        "LocusId": "HTT",
        "LocusStructure": "(CAG)*CAACAG(CCG)*",
        "ReferenceRegion": [
            "chr4:3074876-3074933",
            "chr4:3074939-3074966"
        ],
        "BenignMax": [26, null],
        "PathogenicMin": [36, null]
    }
]
```

The built-in catalog has no thresholds; you need to add your own. Sources for published thresholds include [STRchive](https://strchive.org) (benign, intermediate and pathogenic ranges per locus), [STRipy](https://stripy.org/database) and [GeneReviews](https://www.ncbi.nlm.nih.gov/books/NBK1116/).

Thresholds are compared with the repeat unit count that Abacus reports for that satellite, so they must fit how the satellite is defined in the catalog. Published thresholds are based on how the repeat was counted in the original studies, e.g. whether interruptions or adjacent/partial repeats are included. Where the catalog region or structure differs from that, the published values may need adjusting.

Not supported:
- Thresholds on the combined count of several satellites.
- Pathogenicity determined by repeat motif rather than size (e.g. RFC1, or the `TTTCA` insertions at FAME loci).

## Other Resources

- [gnomAD STR browser](https://gnomad.broadinstitute.org/short-tandem-repeats?dataset=gnomad_r4) — population-level STR variation from gnomAD v4
- [STRipy database](https://stripy.org/database) — curated database of pathogenic STR loci
- [TRexplorer](https://trexplorer.broadinstitute.org) — Tandem Repeat Explorer from the Broad Institute

## Notes on FGF14
FGF14 is a complex locus with multiple haplotypes and many variants. The [provided catalog](./src/abacus/str_catalogs/abacus_catalog.json) includes an `FGF14_complex` entry:

```json
[
    {
        "LocusId": "FGF14_complex",
        "LocusStructure": "(TAGTCATAGTACCCCAA|TAGTCATAGTACCA)*(GVA)*",
        "ReferenceRegion": [
            "chr13:102161565-102161565",
            "chr13:102161565-102161726"
        ]
    }
]
```
`FGF14_complex` includes the possible interruptions `GCA` and `GGA` and combines them into `(GVA)*` (`V` = A/C/G, matching whichever base appears at that position). It also adds a common set of flanking regions that are not present in the hg38 reference: `TAGTCATAGTACCCCAA` and `TAGTCATAGTACCA`, each length-matched so a clean `GAA`-phase repeat always follows immediately after.

The insertion `(TAGTCATAGTACCCCAA)*` is described in the following paper:
https://www.nature.com/articles/s41588-024-01808-5/figures/1

The interruptions `GCA` and `GGA`, the alternative flanking region `TAGTCATAGTACCA` and a lot of additional variation not yet captured by `FGF14_complex` are described in the following paper:
https://www.nature.com/articles/s41467-024-52148-1

## Notes on RFC1
RFC1 has one pentanucleotide repeat (`chr4:39348424-39348485`) but a wide range of observed motifs. `RFC1_complex` only includes motifs whose pathogenicity has been classified - benign, likely benign, likely pathogenic or pathogenic. Motifs of unknown pathogenicity are deliberately left out.

```json
[
    {
        "LocusId": "RFC1_complex",
        "LocusStructure": "(AAAAG|AAAGG|AAGGG|AGGGC|ACAGG|AAAGGG)*",
        "ReferenceRegion": "chr4:39348424-39348485"
    }
]
```

| Motif | Classification | Original source |
|---|---|---|
| `AAAAG` | Benign (reference motif) | Cortese et al. 2019 |
| `AAAGG` | Benign below 500 repeat units, likely pathogenic above | Cortese et al. 2019, Dominik et al. 2023 |
| `AAAGGG` | Likely benign | Abramzon et al. 2021 |
| `AGGGC` | Likely pathogenic | Scriba et al. 2023, Dominik et al. 2023 |
| `AAGGG` | Pathogenic | Cortese et al. 2019 |
| `ACAGG` | Pathogenic | Scriba et al. 2020 |

Classifications are taken from the review by [Delforge et al. 2024](https://doi.org/10.1016/j.neurol.2024.03.006) (Fig. 1), which compiles the original sources above. Note:
- The `AAAGG` 500-unit threshold isn't part of the locus structure; apply it to the reported count.
- The Māori pathogenic configuration `(AAAGG)10-25(AAGGG)exp(AAAGG)4-6` (Beecroft et al. 2020) is covered by the included motifs.
