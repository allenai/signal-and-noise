```sh
git lfs install # .ipynb files are tracked with git lfs! (brew install git-lfs)
pip install -r requirements.txt

# Install scaling law code
git clone -b signal-to-noise https://github.com/allenai/OLMo-ladder deps/OLMo-ladder
pip install -e "deps/OLMo-ladder.[plotting]"

# Install eval code
git clone -b signal-to-noise https://github.com/allenai/oe-eval-internal deps/oe-eval-internal
pip install -e "deps/oe-eval-internal.[all]"
```

Analysis tools for pre-training evaluation

### Quick Start
```sh
git lfs install # .ipynb files are tracked with git lfs! (brew install git-lfs)
pip install -r requirements.txt
mkdir olmo-repos # clone olmo repos here if applicable!

# Download model ladder code
git clone https://github.com/allenai/OLMo-ladder olmo-repos/OLMo-ladder
cd olmo-repos/OLMo-ladder
git checkout datados
pip install -e ".[all]"
```

## Other Features

### Setup custom oe-eval-internal
```sh
# (optional) Download custom oe-eval-internal
git clone git@github.com:allenai/oe-eval-internal.git olmo-repos/oe-eval-internal
cd olmo-repos/oe-eval-internal/
git checkout paraphrase # get current project branch
pip install -e ".[dev]" # --no-deps
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

### Install Custom oe-eval 
```sh
# (for vllm support) install nightly vllm=
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

# sanity check
oe-eval --model pythia-160m --task hellaswag:rc::olmes:full --run-local --output-dir workspace

oe-eval --model pythia-160m --task drop::olmes:full gsm8k::olmes:full jeopardy::olmes:full naturalqs::olmes:full squad::olmes:full triviaqa::olmes:full arc_challenge:rc::olmes:full --run-local --output-dir /Users/dhei/ai2/new-evals/workspace --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/local_testing --limit 20

oe-eval --model pythia-160m --task bbh_boolean_expressions:cot::olmes:full --run-local --output-dir /Users/dhei/ai2/new-evals/workspace --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/local_testing --limit 20

# custom suites
oe-eval --model pythia-160m --task rc_basic::custom_suite --run-local --output-dir workspace
```

### Converting OLMo Checkpoints
```sh
# install olmo
git clone git@github.com:allenai/OLMo.git olmo-repos/OLMo
cd olmo-repos/OLMo
pip install -e .

# install nightly transformers
pip install git+https://github.com/huggingface/transformers

# OLMo 1 models (pre-peteish)
INPUT_DIR=/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded
OUTPUT_DIR=/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded-hf
TOKENIZER_PATH=/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/latest/tokenizer.json
python olmo-repos/OLMo/scripts/convert_olmo_to_hf_new.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --tokenizer_json_path $TOKENIZER_PATH

# OLMo 2 models (post-peteish)
INPUT_DIR=/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded
OUTPUT_DIR=/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded-hf
TOKENIZER_PATH=/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/latest/tokenizer.json
python olmo-repos/OLMo/scripts/convert_olmo2_to_hf.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --tokenizer_json_path $TOKENIZER_PATH

# Convert in a batch
conda activate metaeval
/root/ai2/metaeval/convert_checkpoints_peteish.sh

# Detatch from current session
nohup ./scripts/convert_checkpoints_peteish.sh > /tmp/out.out 2>&1 & tail -f /tmp/out.out

nohup python scripts/download_checkpoints.py > /tmp/out.out 2>&1 & tail -f /tmp/out.out
```
