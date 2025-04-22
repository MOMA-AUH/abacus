from setuptools import find_packages, setup

setup(
    name="abacus",
    version="0.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    entry_points={"console_scripts": ["abacus = abacus.main:app"]},
    test_suite="tests",
    python_requires=">=3.11",
    scripts=["src/abacus/scripts/report_template.Rmd"],
    author="Simon Opstrup Drue",
    author_email="simondrue@gmail.com",
)
