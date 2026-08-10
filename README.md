# Abacus

[![Bioconda](https://img.shields.io/conda/vn/bioconda/abacus-str.svg)](https://anaconda.org/bioconda/abacus-str)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

<img src="./img/logo.png" width="150">

## Description

Abacus analyzes STR (Short Tandem Repeat) data from Long-Read Sequencing. It targets Oxford Nanopore (ONT) data and has also been tested on Pacific Biosciences (PacBio) data. The main goal of Abacus is to provide a user-friendly interface for analyzing STR data and to provide a comprehensive report of the analysis results.

Abacus converts each entry of an STR catalog (JSON) into a graph, then re-maps relevant reads from an aligned BAM/CRAM file onto that graph using [minigraph](https://github.com/lh3/minigraph). Repeat counts per read are derived from the resulting path, and used to cluster reads into haplotypes, and STR alleles are called from these groups.

### Haplotyping algorithm

The haplotyping pipeline proceeds in three stages.

**Stage 1 — Outlier removal (preprocessing)**

Reads are grouped into two clusters (H1 and H2) via a Gaussian mixture model fit to the repeat counts. Two outlier removal steps then clean the read set iteratively, re-estimating model parameters and re-grouping after each removal. At most 1 read is removed per cluster per iteration to prevent over-filtering. Both steps are controlled by `--min-n-length-outlier-detection`, the minimum number of reads per haplotype group required to run outlier detection for that group.

1. **Singleton cluster removal**: a haplotype cluster with exactly one read is treated as an outlier; the read is removed and reads are re-grouped.
2. **Length outlier detection**: for each group (H1, H2, or HOM), the median STR base-pair length is computed across spanning reads. A read is flagged as an outlier if it falls outside both Tukey-fence robust bounds *and* a tolerance window of `median × (1 ± tolerance)`, set via `--length-outlier-tolerance`.

**Stage 2 — Length-based heterozygosity test**

A log-likelihood ratio test (LRT) compares a homozygous model (single Gaussian) against a heterozygous model (two Gaussians). If significant (p < `--heterozygosity-alpha`), reads keep their H1/H2 assignments; otherwise all reads are re-tagged HOM and Stage 3 is attempted.

**Stage 3 — Equal-length sequence split test (backup)**

When Stage 2 is not significant — haplotypes have the same repeat count — Abacus attempts a sequence-level split. A partial-order alignment (POA) MSA is built from all reads, the most variable kmer position is identified, and a binomial test (H₀: p = 0.5) is applied to the two most common kmers at that position:

- **Not significant (p ≥ alpha)**: the observed split is consistent with a 50/50 split — two alleles of equal length but distinct sequence. Reads are tagged H1/H2 by which kmer they carry; reads carrying neither are filtered (`not_split_base`).
- **Significant (p < alpha)**: ratio deviates from 50/50, no split is made, locus is called homozygous.

Controlled by `--equal-length-alpha`.

### Reporting results

Results are saved as an HTML report (STR loci, called alleles, and visualizations) and a VCF file with the genotyping calls. `--add-consensus-to-vcf` and `--add-contracted-consensus-to-vcf` add consensus calls to the VCF output. For large catalogs, report generation can be skipped by omitting the `--report` argument.

## Installation

### Via conda (recommended)
```sh
conda install bioconda::abacus-str
```

### From source
To set up the environment for this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/MOMA-AUH/abacus
    cd abacus
    ```

2. Create and activate the conda environment using the provided `env.yml` file:
    ```sh
    conda env create -f env.yml -n abacus
    conda activate abacus
    ```

3. Install Abacus using `pip`:
    ```sh
    pip install .
    ```

## Basic usage
Core arguments:

- `--bam`: BAM or CRAM file with aligned reads.
- `--ref`: reference FASTA used to align the reads.
- `--report` *(optional)*: HTML output path. Omit to skip report generation (useful for large catalogs).
- `--vcf`: VCF output path for the STR genotyping results.
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
- `--min-str-qual`: minimum mean base quality in the STR region; lower is filtered out. Default: `20`.
- `--tol-str-qual`: tolerance for mean base quality in the STR region. Default: `30`.
- `--min-q10-str-quality`: minimum Q10 base quality in the STR region. Default: `15`.
- `--tol-q10-str-quality`: tolerance for Q10 base quality in the STR region. Default: `30`.
- `--max-error-rate`: max allowed error rate in the STR region; higher is filtered out. Default: `0.01`.
- `--tol-error-rate`: tolerance for error rate in the STR region. Default: `0.005`.
- `--max-ref-divergence`: max allowed reference divergence in the STR region. Default: `0.34`.
- `--length-outlier-tolerance`: fraction of the haplotype median STR base-pair length; reads within `median × (1 ± tolerance)` are always kept, even outside the Tukey-fence bounds. Default: `0.10` (10%).
- `--min-n-qc-filtering`: minimum read calls required to run statistical QC filtering. Default: `10`.
- `--min-n-length-outlier-detection`: minimum spanning reads per haplotype group to run length outlier detection. Default: `5`.

#### Haplotype Parameters
- `--min-haplotyping-depth`: minimum depth per called haplotype; below this the locus is called homozygous. Default: `10`.
- `--heterozygosity-alpha`: sensitivity cutoff for the length-based heterozygosity test. Default: `0.05`.
- `--equal-length-alpha`: sensitivity cutoff for the equal-length sequence split test (backup test for equal-length haplotypes). Default: `0.05`.

#### Coverage Parameters
- `--downsample`: randomly downsample reads per locus above this count; useful for high-coverage data (e.g. PacBio PureTarget). `0` disables downsampling. Default: `1000`.
- `--downsample-seed`: random seed for downsampling, for reproducibility. Default: `42`.

#### Output Options
- `--log-file`: path to the log file. Default: `abacus.log`.
- `--keep-temp-files`: keep temporary analysis files. Default: `False`.
- `--add-consensus-to-vcf`: add consensus calls to the VCF output. Default: `False`.
- `--add-contracted-consensus-to-vcf`: add contracted consensus calls to the VCF output. Default: `False`.

## The STR catalog

The STR catalog is a JSON file describing the STR loci to analyze. Each entry contains:

- `LocusId`: unique identifier for the locus, used to refer to it in the analysis results.
- `LocusStructure`: the locus structure, with each repeat unit in parentheses followed by `*`. E.g. ATXN1: `(CTG)*`. Repeat units can be any length and use [IUPAC](https://en.wikipedia.org/wiki/International_Union_of_Pure_and_Applied_Chemistry) base symbols (e.g. `N`, `Y`), and the structure can include non-repeating flanking/interrupting sequence — e.g. HTT: `(CAG)*CAACAG(CCG)*`, where `CAACAG` interrupts the two repeat units. The `|` operator specifies alternative repeat sequences within a unit, e.g. `(CAG|CAA)*` counts both `CAG` and `CAA` toward the repeat count.
- `ReferenceRegion`: the locus's genomic region(s) in the reference genome, as `chr:start-end`. A list of regions can be given to cover the full structure.

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
        "LocusStructure": "(TAGTCATAGTACCCCAA)*(GAA)*",
        "ReferenceRegion": "chr13:102161565-102161726"
    }
]
```

The insertion `(TAGTCATAGTACCCCAA)*` is described in the following paper:
https://www.nature.com/articles/s41588-024-01808-5/figures/1

A lot of additional variation around the FGF14 locus is discussed in the following paper:
https://www.nature.com/articles/s41467-024-52148-1