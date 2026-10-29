# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
    
## [Unreleased]
### Added
- `AFF3` catalog entry
- `RFC1_complex` catalog entry, covering the RFC1 motifs with a published pathogenicity classification
- `PS` FORMAT field; heterozygous genotypes are now phased
- CI workflow
- Optional `BenignMax`/`PathogenicMin` catalog thresholds: count estimates are highlighted in the report summary table and classified in the new VCF fields `STRSTATUS`, `BENIGNMAX` and `PATHOGENICMIN` ([#28](https://github.com/MOMA-AUH/abacus/issues/28))
- `LPSRUS`/`LPSRUC` INFO fields with the longest pure segment (LPS) of each allele, after [Danzi et al. (2025)](https://doi.org/10.1101/2025.01.06.631535)
- Report summary table header stays visible while scrolling ([#28](https://github.com/MOMA-AUH/abacus/issues/28))
- `REFDIV` INFO field: each allele's divergence from the catalog locus structure, so alleles the catalog motif doesn't describe are visible
### Changed
- **Breaking:** VCF output now follows VCFv4.5 and uses the `<CNV:TR>` tandem repeat representation (see the README for all fields):
  - `POS`/`REF` is the padding base before the repeat; `POS` was previously reported with a reference base it did not match
  - ALT alleles are `<CNV:TR>` instead of `<STRn>`
  - Repeat composition is reported with the reserved `SVLEN`, `RN`, `RUS` and `RUC` INFO fields, splitting the consensus into runs of identical repeat units
  - `REPCN`/`REPCI` are replaced by INFO `TRUC`/`CITRUC`, with the confidence interval given as offsets from the estimate
  - `CONS` moved from FORMAT to INFO
  - `LC` is replaced by the reserved `DP`, and `AD`, `ADSP` and `ADFL` now include the reference allele (always `0`)
  - Fixed the `##reference` header line
- Consensus built by hill-climbing search instead of spoa (new dependency: `rapidfuzz`); consensus length now matches the estimated repeat count ([#25](https://github.com/MOMA-AUH/abacus/issues/25))
- Consensus built from kmers and slot structure; minigraph subprocess removed
- Faster read quality and 5mC probability conversion (native `bytes.translate`, in-place `bytearray` mutation)
- Faster parameter estimation and report
- Faster consensus search when some reads are much noisier than the rest: reads far from the consensus no longer propose changes
- Faster and more accurate consensus search for long alleles
- Consensus search 15x faster at high depth (min-support filter, early-terminate scoring, inline moves)
- Skip report payloads and CSV writes in VCF-only runs
- Refactored locus execution in multiprocessing
- A catalog entry whose `LocusStructure` has no repeat unit now raises an error instead of being skipped with a log message
- **Breaking:** reference divergence is an outlier filter within each haplotype instead of a hard cutoff, so an allele that differs from the catalog motif is kept instead of removed (e.g. an expansion of a different unit). `--max-ref-divergence` is replaced by `--tol-ref-divergence` (default `0.20`)
- `FGF14_complex` catalog entry is now `(TAGTCATAGTACCCCAA|TAGTCATAGTACCA)*(GVA)*`, adding the alternative flanking region `TAGTCATAGTACCA` and the `GGA` interruption
### Fixed
- Consensus mapping failures now retry with denser seeds, then emit `NoCall` instead of crashing
- Report handling and display of 0-copy alleles
- `LPSRUC` is `0` (not `.`) for zero-copy alleles
- VCF genotype and per-allele field ordering
- Kmer chopping for mixed-length alternatives, e.g. `(GAA|GA)*`
- Reading-frame slips: slipped satellite copies are split into short-copy stretches, not packed (e.g. `CG CG` in a `(CGG)*` repeat was counted as 1 `CGCG`, but is now counted as 2 `CG`)
- Insertions at anchor boundaries
- Initial estimates when one allele is under-represented
- Report crashing when no reads are filtered/unmapped ([#26](https://github.com/MOMA-AUH/abacus/issues/26))
- Segfault at loci with many distinct kmers, e.g. long RFC1 expansions ([#29](https://github.com/MOMA-AUH/abacus/issues/29))
- Crash when the equal-length sequence split is cancelled for a singleton cluster; unclassified reads now return to HOM
- Parallel runs of the same sample sharing a temporary directory; it is now named after the VCF instead of the sample ID
- Crash on chrY loci in XX samples (ploidy 0); chrY loci are now skipped when `--sex XX`
- Crash on an empty catalog; a header-only VCF is now written and the report is skipped
- Loci without usable reads were silently missing from the VCF; they now get a `NoReads`/`NoCall` record and a log warning
- A failing locus is now logged with its ID and traceback
- 0-copy alleles (reads that skip the repeat) were filtered as low quality; they are now called with `TRUC=0`
- Empty repeat units where a read doesn't fit the motif (minigraph deletes whole copies and inserts the read's bases elsewhere); such reads are now realigned along their path (new dependency: `biopython`)
- Reads with long unaligned ends beyond short flanks were misclassified as spanning; they are now treated as flanking reads instead of contributing falsely small repeat counts
### Removed
- `--add-contracted-consensus-to-vcf` and the `CONTRCONS` VCF field; `RUS`/`RUC` carry the same information
- VCF fields `END`, `RU`, the unused `INFO/REF` and `FILTER/LowQual`, and the `<STRn>` ALT header lines

## [2.0.0] - 2026-06-29
### Added
- Built-in STR catalog bundled with the package; `--str-catalog` is now optional and defaults to the included abacus catalog
- `--show-catalog` flag to print the built-in catalog JSON to stdout and exit (pipe to a file to use as a starting point for customization)
- `--downsample` and `--downsample-seed` parameters to randomly subsample reads per locus when coverage exceeds a threshold
- A version endpoint ([#14](https://github.com/MOMA-AUH/abacus/issues/14))
- A Changelog ([#13](https://github.com/MOMA-AUH/abacus/issues/13))
- A Dockerfile ([#12](https://github.com/MOMA-AUH/abacus/issues/12))
- Support for CRAM files ([#8](https://github.com/MOMA-AUH/abacus/issues/8))
- Backup heterozygosity detection step when haplotypes share the same repeat count. This splits the reads using differences in the read sequences e.g. allele-specific interruptions.
- Sequence length outlier detection and filtering. The `--length-outlier-tolerance` parameter is used to control the tolerance window around the haplotype median.
- `--threads` argument for multi-threading
- `--loci-subset-file` argument to specify a text file of locus IDs to analyze
### Changed
- HTML report generation is now optional; provide a path with `--report` to produce a report, or omit to skip it (useful for large catalogs)
- Updated path logic to include an OR ("|") operator for satellites. Behind the scenes the graph is now built with networkx for easier maintenance.
- Renamed `str_catalouges` directory to `str_catalogs` (fixing typo)
- Fallback to `loqus_id` if there is only one satellite id, and no `VariantId` ([#16](https://github.com/MOMA-AUH/abacus/issues/16))
- Migrated packaging from `setup.py` to `pyproject.toml`
- Final parameter estimation is now performed using the homozygote restricted to H1, H2, and HOM read groups
- Long flanking reads are now handled correctly in outlier detection
- QC filtering parameters and plot updated
- Report updated to match new features and changes. Summary table for all loci added.
- Strand information added to detailed locus plots and tables
- Updated VCF output parameters for improved accuracy
- Updated built-in STR catalog
- Graph is now built only once per locus for improved performance
### Fixed
- Typo catalog. Normally an interface breaking change, but we are still on semantic version 0.0.x ([#15](https://github.com/MOMA-AUH/abacus/issues/15))
- logpdf handling for `-Inf` values in parameter estimation and parameter estimation bug ([#20](https://github.com/MOMA-AUH/abacus/issues/20))
- Stopped flanking reads from uneccerily changing haplotype group
- SatelliteID not correctly passed in certain configurations
- Bug in Tukey's fence outlier detection
- Sequence splitting causing NA values in the report
- Interruption phasing issue when interruptions were shifted in one allele (increased gap penalty in SPOA alignment)

## [0.0.0]
### Added 
- Highlight of loci with wide confidence intervals
- Trim of low qual end bases
- VCF output
- Haplotype class
### Changed
- Updated report
- Updated README
### Fixed
- No coverage bug
- UIPAC handling bug
