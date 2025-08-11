"""
Models used to compute SNR at 1B, 7B, 13B, 32B scales
"""
SNR_MODELS = {
    "olmo2_1b": {
        "count": 8,
        "models": [
            "meta-llama/Llama-3.2-1B",
            "microsoft/Orca-2-13b",
            "Qwen/Qwen1.5-1.8B",
            "Qwen/Qwen1.5-4B",
            "Qwen/Qwen2-0.5B",
            "Qwen/Qwen2.5-0.5B",
            "deepseek-ai/deepseek-moe-16b-base",
            "huggyllama/llama-7b",
        ],
        "flops_target": 3.6e22,
        "flops_range": (2.1176470588235293e22, 6.12e22),
    },
    "olmo2_7b": {
        "count": 13,
        "models": [
            "meta-llama/Llama-3.2-3B",
            "allenai/OLMo-2-1124-7B",
            "allenai/OLMo-7B-0424-hf",
            "allenai/OLMo-7B-0724-hf",
            "allenai/OLMo-7B-hf",
            "Qwen/Qwen1.5-7B",
            "Qwen/Qwen2.5-1.5B",
            "HuggingFaceTB/SmolLM2-1.7B",
            "01-ai/Yi-1.5-6B",
            "01-ai/Yi-1.5-9B",
            "01-ai/Yi-6B",
            "01-ai/Yi-9B",
            "huggyllama/llama-30b",
        ],
        "flops_target": 1.68e23,
        "flops_range": (9.88235294117647e22, 2.8559999999999996e23),
    },
    "olmo2_13b": {
        "count": 6,
        "models": [
            "allenai/OLMo-2-1124-13B",
            "Qwen/Qwen2-7B",
            "Qwen/Qwen2.5-3B",
            "01-ai/Yi-34B",
            "huggyllama/llama-30b",
            "huggyllama/llama-65b",
        ],
        "flops_target": 3.9e23,
        "flops_range": (2.2941176470588235e23, 6.63e23),
    },
    "olmo2_32b": {
        "count": 9,
        "models": [
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Meta-Llama-3-8B",
            "Qwen/Qwen1.5-32B",
            "Qwen/Qwen1.5-72B",
            "Qwen/Qwen2.5-14B",
            "Qwen/Qwen2.5-7B",
            "01-ai/Yi-1.5-34B",
            "deepseek-ai/deepseek-llm-67b-base",
            "microsoft/phi-4",
        ],
        "flops_target": 1.15e24,
        "flops_range": (6.764705882352941e23, 1.955e24),
    },
}
