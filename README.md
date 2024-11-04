<h1 align="center">Abacus</h1>
<p align="center"><i>A tool for analyzing Short Tandem Repeats (STRs) from Long-Read Sequencing data</i></p>

<p align="center">
    <img width="250px" src="./img/logo.png">
</p>

## Description

Abacus is a tool for analyzing STR (Short Tandem Repeat) data from Long-Read Sequencing technologies. It is designed to work with data from the Oxford Nanopore Technologies (ONT) platform, but have also been tested with data from the Pacific Biosciences (PacBio) platform. The main goal of Abacus is to provide a user-friendly interface for analyzing STR data, and to provide a comprehensive report of the analysis results. 

Abacus works first by converting the entries of an STR catalogue (JSON) into graphs, which are then used to analyze the STR data from an aligned BAM file. Each read in the BAM file is first mapped to the graph using [minigraph](https://github.com/lh3/minigraph).

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
    - [The STR catalouge](#the-str-catalouge)

## Installation
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
To run Abacus, you need to provide the following arguments:

- `--sample_id`: The identifier of the sample that you are analyzing.
- `--bam`: The path to the BAM file that contains aligned reads from the Long-Read Sequencing data.
- `--ref`: The path to the reference FASTA file that was used to align the reads in the BAM file.
- `--str_catalogue`: The path to the STR catalogue (JSON) that contains the information about the STR loci that you want to analyze.
- `--report`: The path to the HTML file where the analysis results will be saved.

The following command shows an example of how to run Abacus:
```sh
abacus --sample_id sample1 --bam input.bam --ref reference.fa --str_catalogue str_catalogue.json --report output.html
```

After running the command, Abacus will analyze the STR data from the BAM file and save the results in the HTML report file.

### The STR catalouge

The STR catalogue is a JSON file that contains information about the STR loci that you want to analyze. Each entry in the catalogue should contain the following information:

- `LocusId`: The identifier of the STR locus. This can be any string that uniquely identifies the locus. Is used to refer to the locus in the analysis results.
- `LocusStructure`: The structure of the STR locus, where each repeat unit is enclosed in parentheses and followed by an asterisk. For example, the structure of the ATXN1 locus: `(CTG)*`. The structure can contain any number of repeat units of any length and can contain [IUPAC](https://en.wikipedia.org/wiki/International_Union_of_Pure_and_Applied_Chemistry) base symbols, such as `N` or `Y`. The structure can also contain non-repeating sequences, such as flanking regions or interruptions. For example, the structure of the HTT locus: `(CAG)*CAACAG(CCG)*`, where `CAACAG` is a non-repeating sequence.
- `ReferenceRegion`: The genomic region of the STR locus in the reference genome. This can be a single region of the entire structure of the STR locus, or a list of regions that cover the entire structure of the STR locus. The regions should be in the format `chr:start-end`, where `chr` is the chromosome name and `start` and `end` are the start and end positions of the region, respectively.

In `str_catalouges/moma_repeat_variants_catalogue_240521.json` you can find a comprehensive list of STRs that are known to be variable in the human genome, which can be used as a starting point for your analysis. You can also create your own STR catalogue by following the format described above. Underneath is an example of the structure of the STR catalogue:

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