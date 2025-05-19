# AutoBencher

This is a re-factor of the original [AutoBencher repo](https://github.com/XiangLi1999/AutoBencher), to generate the "AutoBencher" dataset in the paper.

### Quick Start
To quick start, you need to install the following dependencies:
```bash
pip insall -r requirements.txt 
```

Then, you can run the following command to start the benchmark to experiment with the kowledge intensive tasks:
```bash 
python run_script.py wiki
python run_script.py multilingual
python run_script.py math
```

### Steps to Reproduce
To reproduce the construction of the 33K AutoBench dataset, use the following:
    
```bash
# Command for AutoBench KnowledgeQA (each produces ~1.6K questions for ~$0.50 or 4M generated tokens):
python wiki_autobencher.py --exp_mode autobencher --test_taker_modelname gpt-4o-mini --use_helm no --agent_modelname gpt-4o-mini --theme history --outfile_prefix1 KI/history.

# Execute generation commands in parallel:
parallel --jobs 10 < ../run_autobencher_kl.sh

# Command for AutoBench Math (not used)
python math_autobencher.py --exp_mode autobencher --test_taker_modelname gpt-4o-mini --use_helm no --agent_modelname gpt-4o-mini --outfile_prefix1 MATH/.
```