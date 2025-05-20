Code for "Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation", currently under submission.

### Quick Start

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

Note: This supplimentary submission already comes with all the code. The datasets are too large to submit under the 100MB limit supplimentary material, this is hosted as follows. As-per the NeurIPS guidelines for large datasets (https://neurips.cc/Conferences/2025/DataHostingGuidelines), we published an anonymized URL in the Harvard Dataverse, which is accessible here: 

- https://dataverse.harvard.edu/previewurl.xhtml?token=d5947913-fbdd-4a06-8876-1fd2fdead31f

Once downloaded, the resulting tree of data should appear as follows:

```
analysis/data
├── benchmarks.parquet
├── consistent_ranking.parquet
├── consistent_ranking_small.parquet
└── olmo2_random_seeds.parquet
```