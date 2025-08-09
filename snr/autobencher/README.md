Code to generate Autobencher, based on the [original repo](https://github.com/XiangLi1999/AutoBencher). The dataset is available here: https://huggingface.co/datasets/allenai/autobencher-qa-33k.

```sh
pip install -r requirements.txt
```

### 1. Generate questions
To reproduce the construction of the 33K AutoBench dataset, use the following:
    
```bash
# Command for AutoBench KnowledgeQA (each produces ~1.6K questions for ~$0.50 or 4M generated tokens):
python wiki_autobencher.py --exp_mode autobencher --test_taker_modelname gpt-4o-mini --use_helm no --agent_modelname gpt-4o-mini --theme history --outfile_prefix1 KI/history.

# Execute generation commands in parallel:
parallel --jobs 10 < ../run_autobencher_kl.sh
```

### 2. Generate distractors

```sh
python compile_autobencher.py
```