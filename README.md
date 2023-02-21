# treetap

Generating Explainable Rule Sets from Tree-Ensembles by Answer Set Programming.

This repository contains the source code for the paper 
"Generating Global and Local Explanations for Tree-Ensemble Learning Methods by Answer Set Programming".

## Example

Please see the [demo notebook](demo.ipynb) for a more detailed example.

Given a trained tree-ensemble model, like this:
```python
rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=SEED, n_jobs=1)
rf.fit(x_train, y_train)
```

Fit Rule Extractor for your Trees and Explanation type:
```python
# for global explanations
rf_global_extractor = RFGlobalRuleExtractor()
rf_global_extractor.fit(x_train, y_train, model=rf, feature_names=feat)
```

(Optional: Check the base rules in the trees with the following)
```python
# this returns a pandas dataframe
rf_global_extractor.export_rule_df()
```

Analyze the rules with ASP
```python
global_res_str = rf_global_extractor.transform(x_train, y_train)
with open('./tree_asp/tmp/scratch/rules.lp', 'w', encoding='utf-8') as outfile:
    outfile.write(global_res_str)
with open('./tree_asp/tmp/scratch/class.lp', 'w', encoding='utf-8') as outfile:
    outfile.write('class(1).'.format(int(y_train.nunique() - 1)))
# solving with ASP
o = subprocess.run(['clingo', 
                    './tree_asp/asp_encoding/global_accuracy_coverage.lp', 
                    './tree_asp/tmp/scratch/rules.lp',
                    './tree_asp/tmp/scratch/class.lp', 
                    '0', '--parallel-mode=8,split'], 
                   capture_output=True, timeout=600)
answers, clasp_info = generate_answers(o.stdout.decode())
# printing the results
for ans_set in answers:
    if not ans_set.is_optimal:
        continue
    else:
        for ans in ans_set.answer:   # list(tuple(str, tuple(int)))
            pat_idx = ans[-1][0]
            pat = rf_global_extractor.rules_[pat_idx]  # type: Rule
            print(f'class {pat.predict_class} IF {pat.rule_str}')
        break
```

... which should print out something like:
```
class 1 IF occupation_Handlers_cleaners <= 0.5 AND sex_Female <= 0.5 AND occupation_Machine_op_inspct <= 0.5 AND marital_status_Married_civ_spouse > 0.5 AND hours_per_week > 33.5
```

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

You may have difficult time building the container and running the scripts on M1/M2-based Macs.   
The scripts have only been tested on a standard Ubuntu machine and Intel Mac.

# Publications/Citations

* Akihiro Takemura, Katsumi Inoue:
Generating Explainable Rule Sets from Tree-Ensemble Learning Methods by Answer Set Programming. ICLP Technical Communications 2021: 127-140 https://arxiv.org/abs/2109.08290


# License
TBD.
