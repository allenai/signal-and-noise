### Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation

<p align="center">
  <a href="https://github.com/allenai/signal-and-noise/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/badge/license-Apache 2.0-green">
  </a>
  <a href="">
    <img alt="Paper URL" src="https://img.shields.io/badge/paper-arxiv-red">
  </a>
  <a href="https://huggingface.co/datasets/allenai/signal-and-noise">
    <img alt="Huggingface URL" src="https://img.shields.io/badge/data-huggingface-yellow">
  </a>
</p>

Our work studies the ratio between *signal*, a benchmark's ability to separate models; and *noise*, a benchmark's sensitivity to random variability during training steps. 

**Setup**

```sh
git clone https://github.com/allenai/signal-and-noise
pip install -e .
```

### Calculating SNR

Our core signal to noise calculation can be produced in a few lines. Given a scores from a population of models (`signal_scores`) and intermediate checkpoints (`noise_scores`), pseudocode is as follows:

```python
import numpy as np

def signal_to_noise_ratio(signal_scores: np.ndarray, noise_scores: np.ndarray) -> float:
    """
    signal = \max_{j,k} |m_j - m_k| / m̄
    noise = σ_m / m̄
    snr = signal / noise
    """"
    dispersion = np.max([np.abs(mj - mk) for mj in signal_scores for mk in signal_scores])
    signal = dispersion / np.mean(signal_scores)
    noise = np.std(noise_scores) / np.mean(noise_scores)
    snr = signal / noise
    return snr
```

---

### Using the evaluation dataset

Pull all the model evaluations used in this project (from [huggingface.co/datasets/allenai/signal-and-noise](https://huggingface.co/datasets/allenai/signal-and-noise)):

```python
import pandas as pd
from snr.download.hf import pull_predictions_from_hf

local_path = pull_predictions_from_hf("allenai/signal-and-noise", split_name='core')
df = pd.read_parquet(local_path)

print(f'Loaded {len(df):,} model evaluations')
>>> Loaded 388,924 model evaluations
```

---

### Evaluating a benchmark

**@davidheineman TODO: How to evaluate YOUR benchmark with our tool (maybe with minieval?)**

1. Running and calculating decision accuracy (with HF models)
2. Running and fitting prediction error (with HF models)
3. Running and calculating SNR (with HF models)

**One way to get more usage is put together a quick demo!**

---

### Reproducing tables & figures

The [`analysis/`](./analysis/) folder contains notebooks to reproduce the core findings of our work. Here is a brief description of each:

```sh
── analysis
   ├── quick_start.ipynb     # Demo analysis notebook for our results
   ├── datadecide.ipynb      # (Sec. 1, 3 + Appendix) Corr. between SNR and decision accuracy
   ├── scaling.ipynb         # (Sec. 3 + Appendix)    Corr. between SNR and scaling laws
   ├── table.ipynb           # (Sec. 5) Intervention: Average last n checkpoints to reduce SNR
   ├── smooth_last_n.ipynb   # (Sec. 5) Intervention: Average checkpoints when early stopping to reduce SNR
   ├── smooth_metric.ipynb   # (Sec. 5) Intervention: Track BPB to reduce SNR
   ├── smooth_subtasks.ipynb # (Sec. 5) Intervention: Filter subtasks by their SNR
   ├── sample_size.ipynb     # (Appendix) Reducing sample size
   ├── snr_variants.ipynb    # (Appendix) Alternative measures for signal and noise
```

### Citation

```
TODO
```

<!-- ```sh
mkdir deps # directory for olmo repos

# Install scaling law code
git clone -b signal-to-noise https://github.com/allenai/OLMo-ladder deps/OLMo-ladder
cd deps/OLMo-ladder
pip install -e ".[all]"

# Install eval code
git clone -b signal-to-noise https://github.com/allenai/oe-eval-internal deps/oe-eval-internal
pip install -e "deps/oe-eval-internal.[all]"

# Download seed / data order evals
python analysis/utils/comet_utils.py --workspace ai2 --project olmo2-model-ladder-davidh --output-dir analysis/data/random_seeds --output-filename olmo2_random_seeds.csv
```

---

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
``` -->
