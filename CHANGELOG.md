# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
    
## [Unreleased]
### Added
- A version endpoint (#14)
- A Changelog (#13)
- A Dockerfile (#12)
### Changed
- Fallback to `loqus_id` if there is only one satellite id, and no `VariantId` (#16)
### Fixed
- Typo catalog. Normally an interface breaking change, but we are still on semantic version 0.0.x (#15)

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