from pathlib import Path

REPORT_TEMPLATE = Path(__file__).parent / "report.Rmd"

# Link: https://www.dnabaser.com/articles/IUPAC%20ambiguity%20codes.html
AMBIGUOUS_BASES_DICT = {
    "N": ["A", "T", "C", "G"],
    "R": ["A", "G"],
    "Y": ["T", "C"],
    "S": ["G", "C"],
    "W": ["A", "T"],
    "K": ["G", "T"],
    "M": ["A", "C"],
    "B": ["C", "G", "T"],
    "D": ["A", "G", "T"],
    "H": ["A", "C", "T"],
    "V": ["A", "C", "G"],
}
