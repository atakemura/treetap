# treetap

This repository contains the source code for "Generating Global and Local Explanations for Tree-Ensemble Learning Methods by Answer Set Programming".

# Project Structure
```text
PROJECT_ROOT
| datasets
|   datasets            // includes formatted datasets
|   ingestion.py        // converting raw datasets into formatted datasets
|
| docker
|   Dockerfile          // container information for reproducibility
|
| tree_asp
|   asp_encoding        // includes ASP encoding for rule set generation
|       ...
|   rule_extractor.py   // candidate rule extractor module
|   ...
|
| hyperparameter.py     // hyperparameter optimization settings for all experiments
| run_benchmark.py      // runs 'vanilla' versions of algorithms, with hyperparameter optimization
| run_experiments.py    // runs the rule set generation methods, with hyperparameter optimization
| run_weka.py           // runs RIPPER with hyperparameter optimization
| ...

```

# Experiments

The easiest way to reproduce the experiments are:
* download the repository
* run `docker build` using the provided Dockerfile
* attach to the container e.g., `docker exec -it [CONTAINER NAME] /bin/bash`
* `su - user` (by default it will log you in as root)
* add Python, conda etc. to PATH `export PATH=/opt/conda/bin:/opt/conda/condabin:/opt/conda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH`
* Before running the scripts below, create empty directories with paths mentioned in the scripts, e.g., `mkdir -p tmp/journal/local tmp/journal/global` etc. If you see a FileNotFoundError during experiments this is the likely cause.
* `bash run_all.sh` or run individual experiments like `python run_benchmark.py` etc.

# Dependencies

(If you use the provided Dockerfile, all the following dependencies are already installed on your container.)

If you wish to just use this method for global and local explanations, you need the following:
* clingo (including Python package, see https://potassco.org/doc/start/)
* scikit-learn
* LightGBM
* pandas
* numpy

Additionally, if you wish to reproduce the experiments, you need the following:
* Weka (including Python package, requires OpenJDK11)
* optuna
* RuleFit

# Known Issues

You may have difficult time building the container and running the scripts on M1/M2-based Macs. The scripts have only been tested on a standard Ubuntu machine and Intel Mac.

# Publications/Citations
n/a.

# License
TBD.
