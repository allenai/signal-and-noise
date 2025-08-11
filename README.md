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

<!-- ```sh
mkdir deps # directory for olmo repos

# Install scaling law code
git clone -b signal-to-noise https://github.com/allenai/OLMo-ladder deps/OLMo-ladder
cd deps/OLMo-ladder
pip install -e ".[all]"
``` -->

**Quick Start**

```sh
python snr/snr_table.py
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
    """
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

**Utilities for handling eval results**

```python
# Use get_slice() to get specific results
from snr.dataloader import get_slice
df_subset = get_slice(df, model='OLMo-2-1124-13B', task=['arc_challenge', 'arc_easy'])

print(df_subset[['task', 'primary_score']])
>>>          task  primary_score
>>> arc_challenge       0.639932
>>>      arc_easy       0.884259

# Use get_nd_array() to get a numpy array of results
from snr.dataloader import get_nd_array
tasks, arr = get_nd_array(df, col='task', metric='primary_score', model='OLMo-2-1124-13B', task=['arc_challenge', 'arc_easy'])

print(arr)
>>> [0.63993174 0.88425926]
```

<details>
<summary>Compute decision accuracy</summary>

```python
from snr.dataloader import get_slice
from snr.metrics import decision_acc_fast

scores_small  = get_slice(df, size='150M', task='arc_easy', step=38157)
scores_target = get_slice(df, size='1B', task='arc_easy', step=69369)

decision_acc = decision_acc_fast(
    scores_small = scores_small.sort_values('model')['primary_score'],
    scores_target = scores_target.sort_values('model')['primary_score']
)

print(decision_acc)
>>> 0.93
```

</details>

<details>
<summary>Compute scaling law error</summary>

```python
from snr.ladder_wrapper import run_ladder
from snr.constants.ladder import LADDER_MODEL_NAMES

_, _, (error_7b, error_13b) = run_ladder(
    df,
    task='arc_easy',
    train_models=LADDER_MODEL_NAMES,
    eval_models=["peteish7", "peteish13-highlr"]
)

print(error_7b, error_13b)
>>> 0.0398 0.0553
```

</details>

<details>
<summary>Compute signal-to-noise ratio</summary>

For models < 1B, we use the DataDecide data to compute SNR:

```python
import numpy as np
from snr.metrics import signal_to_noise_ratio

scores_df = get_slice(df, size='150M', task='arc_easy').sort_values('step')

# numpy array of scores in shape (mix, checkpoint)
scores_arr = np.array([lst for lst in scores_df.groupby('mix')['primary_score'].apply(list)])

signal = [np.mean(scores) for scores in scores_arr]
noise  = scores_arr.flatten()

snr = signal_to_noise_ratio(signal, noise)

print(snr)
>>> 3.389
```

For models > 1B, we use the external model scores to compute SNR:

```python
from snr.constants.signal import SNR_MODELS
from snr.metrics import signal_to_noise_ratio

signal_models = SNR_MODELS['olmo2_13b']['models']

noise_df = get_slice(df, model='peteish13-highlr', task=task)
signal_df = df[df['model_path'].isin(signal_models) & (df['task'] == task)]

signal = list(signal_df['primary_score'])
noise  = list(noise_df.sort_values('step')['primary_score'])[-30:]

snr = signal_to_noise_ratio(signal, noise)

print(snr)
>>> 169.776
```

</details>

---

### Evaluating a benchmark

We include the models used in our analysis in [snr/constants/models.py](snr/constants/models.py). They are organized by their huggingface `model` and `revision`.

```python
# 225 DataDecide models (for decision accuracy)
from snr.constants.models import MODEL_LIST_DATADECIDE_FINAL

print(MODEL_LIST_DATADECIDE_FINAL[0])
>>> {'model': 'allenai/DataDecide-c4-150M', 'revision': 'main'}

# Scaling law models (for prediction error)
from snr.constants.models import MODEL_LADDER_LIST

print(MODEL_LADDER_LIST[0])
>>> {'model': 'allenai/OLMo-Ladder-190M-0.5xC', 'revision': 'main'}

# Signal and noise models (for signal-to-noise ratio)
from snr.constants.signal import SNR_MODELS
from snr.constants.models import MODEL_LIST_FINAL_30_1B, MODEL_LIST_FINAL_30_7B, MODEL_LIST_FINAL_30_13B, MODEL_LIST_FINAL_30_32B

print(MODEL_LIST_FINAL_30_1B[0])
>>> {'model': 'allenai/OLMo-2-0425-1B', 'revision': 'stage1-step1610000-tokens3377B'}
```

Our evaluation used [OLMES](https://github.com/allenai/olmes). To install the eval infrastructure:

```sh
git clone https://github.com/allenai/olmes.git deps/olmes
cd deps/olmes
pip install -e ".[all]"
```

A list of all task aliases we used in this work is in [`snr/scripts/oe_eval_tasks.py`](./snr/scripts/oe_eval_tasks.py)

```python
from snr.scripts.oe_eval_tasks import RC_TASKS_OLMES

print(RC_TASKS_OLMES)
>>> ["arc_challenge:rc::olmes:full", "arc_easy:rc::olmes:full", "boolq:rc::olmes:full", ...]
```

Then, use launch with this run command:

```sh
# Run eval on a model / revision pair from HF
oe-eval \
  --model allenai/OLMo-2-0425-1B \
  --revision stage1-step1610000-tokens3377B \
  --task arc_challenge:rc::olmes:full \
  --model-type vllm \
  --gpus 1
```

We include an example script to mass-launch evals in [`snr/scripts/launch_eval.py`](./snr/scripts/launch_eval.py).

Then, to compute decision accuracy, scaling laws and SNR, see the previous section!

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
