FROM continuumio/miniconda3

WORKDIR /app

# Create abacus conda environment
COPY env.yml .
RUN conda env create -f env.yml -n abacus

# Make RUN commands use the abacus conda environment
SHELL ["conda", "run", "-n", "abacus", "/bin/bash", "-c"]

# Install the python app
COPY . /app
RUN pip install .

# Set abacus to run in conda env as entrypoint for the container
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "abacus", "abacus"]
