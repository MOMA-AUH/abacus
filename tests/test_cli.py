import base64
import json
import random
import subprocess
from importlib.metadata import version
from importlib.resources import files

import pysam


def test_installed_cli_writes_report_and_matching_serial_parallel_calls(tmp_path):
    assert version("abacus") in subprocess.check_output(["abacus", "--version"], cwd=tmp_path, text=True, timeout=30)
    assert json.loads(subprocess.check_output(["abacus", "--show-catalog"], cwd=tmp_path, text=True, timeout=30))
    subprocess.run(["abacus", "--help"], cwd=tmp_path, check=True, timeout=30)

    rng = random.Random(42)
    cases = [("chr1", "GAA", 20), ("chr2", "CAG", 10)]
    sequences = {
        chrom: "".join(rng.choices("ACGT", k=600)) + motif * count + "".join(rng.choices("ACGT", k=600))
        for chrom, motif, count in cases
    }
    reference = tmp_path / "reference.fa"
    reference.write_text("".join(f">{chrom}\n{sequence}\n" for chrom, sequence in sequences.items()))
    pysam.faidx(str(reference))
    bam = tmp_path / "reads.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": chrom, "LN": len(seq)} for chrom, seq in sequences.items()]}
    with pysam.AlignmentFile(str(bam), "wb", header=header) as output:
        for tid, (chrom, sequence) in enumerate(sequences.items()):
            for number in range(8):
                read = pysam.AlignedSegment()
                read.query_name = f"{chrom}_read_{number}"
                read.query_sequence = sequence
                read.flag = 16 if number % 2 else 0
                read.reference_id = tid
                read.reference_start = 0
                read.mapping_quality = 60
                read.cigar = [(0, len(sequence))]
                read.query_qualities = pysam.qualitystring_to_array("I" * len(sequence))
                output.write(read)
    pysam.index(str(bam))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([
        {"LocusId": f"SMOKE_{chrom}", "LocusStructure": f"({motif})*", "ReferenceRegion": f"{chrom}:600-{600 + len(motif) * count}"}
        for chrom, motif, count in cases
    ]))

    records = []
    for threads in (1, 2):
        vcf = tmp_path / f"output-{threads}.vcf"
        command = [
            "abacus", "--bam", str(bam), "--ref", str(reference), "--str-catalog", str(catalog),
            "--sample-id", "dependency_smoke", "--vcf", str(vcf), "--threads", str(threads),
            "--log-file", str(tmp_path / f"run-{threads}.log"),
        ]
        if threads == 1:
            command += ["--report", str(tmp_path / "report.html")]
        subprocess.run(command, cwd=tmp_path, check=True, timeout=180)
        rows = [line for line in vcf.read_text().splitlines() if not line.startswith("#")]
        assert len(rows) == 2
        for row in rows:
            fields = row.split("\t")
            assert fields[6] == "PASS"
            assert fields[9].split(":")[0] == "1/1"
            info = dict(field.split("=", 1) for field in fields[7].split(";") if "=" in field)
            expected_count = 20 if fields[0] == "chr1" else 10
            assert abs(float(info["TRUC"]) - expected_count) < 0.1
        records.append(sorted(rows))
    assert records[0] == records[1]

    report = (tmp_path / "report.html").read_text()
    assert "<html" in report.lower()
    assert "dependency_smoke" in report
    logo = files("abacus").joinpath("img/logo.png").read_bytes()
    assert base64.b64encode(logo).decode() in report
