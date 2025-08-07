### Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation

<p align="center">
  <a href="https://github.com/allenai/signal-and-noise/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/signal-and-noise">
  </a>
  <a href="">
    <img alt="Paper URL" src="https://img.shields.io/badge/paper-link-blue">
  </a>
</p>

Code and data for reproducing results in the signal and noise paper.

```sh
pip install -r requirements.txt

# Install the custom version of https://github.com/allenai/OLMo-ladder
cd deps/OLMo-ladder
pip install -e ".[plotting]"

# Install eval library (not necessary for analysis code)
git clone https://github.com/allenai/olmes deps/olmes
cd deps/olmes
pip install -e ".[all]"
```

### Quick Start

```sh
git lfs install # .ipynb files are tracked with git lfs! (brew install git-lfs)
pip install -r requirements.txt
mkdir deps # directory for olmo repos

# Install scaling law code
git clone -b signal-to-noise https://github.com/allenai/OLMo-ladder deps/OLMo-ladder
pip install -e "deps/OLMo-ladder.[plotting]"

# Install eval code
git clone -b signal-to-noise https://github.com/allenai/oe-eval-internal deps/oe-eval-internal
pip install -e "deps/oe-eval-internal.[all]"

# Download seed / data order evals
python analysis/utils/comet_utils.py --workspace ai2 --project olmo2-model-ladder-davidh --output-dir analysis/data/random_seeds --output-filename olmo2_random_seeds.csv
```

### Download Model Ladder Data
```sh
# Download wandb logs (see OLMo library for all downloads)
python olmo/scaling/scaling_laws/download_wandb_logs.py -n 'ai2-llm/olmo-ladder/peteish-moreeval-3B-1xC' -y validation-and-downstream-v2 -o scripts/scaling/data/peteish-moreeval/3B-1xC.csv
python olmo/scaling/scaling_laws/download_wandb_logs.py -n 'ai2-llm/olmo-ladder/peteish-moreeval-3B-2xC' -y validation-and-downstream-v2 -o scripts/scaling/data/peteish-moreeval/3B-2xC.csv
python olmo/scaling/scaling_laws/download_wandb_logs.py -n 'ai2-llm/olmo-ladder/peteish-moreeval-3B-5xC' -y validation-and-downstream-v2 -o scripts/scaling/data/peteish-moreeval/3B-5xC.csv
python olmo/scaling/scaling_laws/download_wandb_logs.py -n 'ai2-llm/olmo-ladder/peteish-moreeval-3B-10xC' -y validation-and-downstream-v2 -o scripts/scaling/data/peteish-moreeval/3B-10xC.csv

# Sanity check: Run variance analysis + predictions
python scripts/scaling/variance_analysis.py -k v2_main_variance -c scripts/scaling/final_variance.json -o figure/peteish-moreeval/variance.pdf --last_n_points 10 --run_prediction
python scripts/scaling/step2.py -k v2_main -c scripts/scaling/step2.json -o figure/peteish-moreeval/step2_main.pdf --skip_perc 0.1 --moving_avg 5
```

### Launching & Processing Evals
```sh
python scripts/launch_evals.py # launch evals on beaker
python analysis/download/aws.py # sync from s3
python analysis/download/preprocess.py # convert to .parquet

# Detatch from current session
nohup python scripts/launch_eval.py > /tmp/out.out 2>&1 & tail -f /tmp/out.out
nohup python analysis/download/aws.py > /tmp/out.out 2>&1 & tail -f /tmp/out.out

# (in case I need it)
nohup python analysis/download/preprocess.py > /tmp/out.out 2>&1 & tail -f /tmp/out.out
nohup python analysis/download/hf.py > /tmp/out.out 2>&1 & tail -f /tmp/out.out
nohup python scripts/download_checkpoints.py > /tmp/out.out 2>&1 & tail -f /tmp/out.out
nohup python scripts/weight_merging/merge.py > /tmp/merge.out 2>&1 & tail -f /tmp/merge.out

# Convert checkpoints
nohup ./scripts/convert_checkpoints_peteish.sh > /tmp/out.out 2>&1 & tail -f /tmp/out.out
```
