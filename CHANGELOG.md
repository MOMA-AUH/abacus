# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
    
## [Unreleased]

## [2.0.0]
### Added
- Built-in STR catalog bundled with the package; `--str-catalog` is now optional and defaults to the included abacus catalog
- `--show-catalog` flag to print the built-in catalog JSON to stdout and exit (pipe to a file to use as a starting point for customization)
- `--downsample` and `--downsample-seed` parameters to randomly subsample reads per locus when coverage exceeds a threshold
- A version endpoint (#14)
- A Changelog (#13)
- A Dockerfile (#12)
- Support for CRAM files (#8)
- Backup heterozygosity detection step when haplotypes share the same repeat count. This splits the reads using differences in the read sequences e.g. allele-specific interruptions.
- Sequence length outlier detection and filtering. The `--length-outlier-tolerance` parameter is used to control the tolerance window around the haplotype median.
- `--threads` argument for multi-threading
- `--loci-subset-file` argument to specify a text file of locus IDs to analyze
### Changed
- HTML report generation is now optional; provide a path with `--report` to produce a report, or omit to skip it (useful for large catalogs)
- Updated path logic to include an OR ("|") operator for satellites. Behind the scenes the graph is now built with networkx for easier maintenance.
- Renamed `str_catalouges` directory to `str_catalogs` (fixing typo)
- Fallback to `loqus_id` if there is only one satellite id, and no `VariantId` (#16)
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
- Typo catalog. Normally an interface breaking change, but we are still on semantic version 0.0.x (#15)
- logpdf handling for `-Inf` values in parameter estimation and parameter estimation bug (#20)
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