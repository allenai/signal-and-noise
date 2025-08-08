# Category coloring for plotting
TASK_CATEGORIES = {
    'hellaswag': 'language',
    'winogrande': 'language',

    'arc_challenge': 'knowledge',
    'arc_easy': 'knowledge', 
    'boolq': 'knowledge',
    'csqa': 'knowledge',
    'openbookqa': 'knowledge',
    'piqa': 'knowledge',
    'socialiqa': 'knowledge',
    'drop': 'knowledge',
    'jeopardy': 'knowledge',
    'squad': 'knowledge', 
    'triviaqa': 'knowledge',
    'olmes_core9': 'knowledge',
    'mmlu': 'knowledge',
    'olmes_core9_mc': 'knowledge',
    'mmlu_mc': 'knowledge',
    'olmes_gen': 'knowledge',
    'autobencher': 'knowledge',
    'autobencher:mc': 'knowledge',
    'mmlu_pro': 'knowledge',
    'agi_eval': 'knowledge',
    'medmcqa': 'knowledge',

    'gsm8k': 'math',
    'minerva': 'math',
    'minerva_math_algebra': 'math',
    'minerva_math_counting_and_probability': 'math',
    'minerva_math_geometry': 'math',
    'minerva_math_intermediate_algebra': 'math',
    'minerva_math_number_theory': 'math',
    'minerva_math_prealgebra': 'math',
    'minerva_math_precalculus': 'math',
    'gsm_plus': 'math',
    'gsm_symbolic_main': 'math',
    'gsm_symbolic_p1': 'math',
    'minerva_math_500': 'math',
    'gsm_symbolic_p2': 'math',

    'mbpp': 'code',
    'mbppplus': 'code',
    'codex_humaneval': 'code',
    'codex_humanevalplus': 'code',

    'bbh': 'reasoning',
    
    'paloma_c4_en': 'loss',
    'paloma_m2d2_s2orc_unsplit': 'loss',

    # Pertubed benchmarks
    'hellaswag:distractors': 'language:distractors',
    'winogrande:distractors': 'language:distractors',
    'hellaswag:para': 'language:para',
    'winogrande:para': 'language:para',
    'hellaswag:enlarge': 'language:enlarge',
    'winogrande:enlarge': 'language:enlarge',

    'arc_challenge:distractors': 'knowledge:distractors',
    'arc_easy:distractors': 'knowledge:distractors', 
    'boolq:distractors': 'knowledge:distractors',
    'csqa:distractors': 'knowledge:distractors',
    'openbookqa:distractors': 'knowledge:distractors',
    'piqa:distractors': 'knowledge:distractors',
    'socialiqa:distractors': 'knowledge:distractors',
    'arc_challenge:para': 'knowledge:para',
    'arc_easy:para': 'knowledge:para', 
    'boolq:para': 'knowledge:para',
    'csqa:para': 'knowledge:para',
    'openbookqa:para': 'knowledge:para',
    'piqa:para': 'knowledge:para',
    'socialiqa:para': 'knowledge:para',
    'arc_challenge:enlarge': 'knowledge:enlarge',
    'arc_easy:enlarge': 'knowledge:enlarge', 
    'boolq:enlarge': 'knowledge:enlarge',
    'csqa:enlarge': 'knowledge:enlarge',
    'openbookqa:enlarge': 'knowledge:enlarge',
    'piqa:enlarge': 'knowledge:enlarge',
    'socialiqa:enlarge': 'knowledge:enlarge',
}
CATEGORIES = set(TASK_CATEGORIES.values())

CATEGORY_COLORS = {
    'language': '#2ecc71',
    'language:enlarge': '#27ae60',
    'language:para': '#1abc9c',
    'language:distractors': '#16a085',
    'knowledge': '#3498db', 
    'knowledge:enlarge': '#2980b9',
    'knowledge:para': '#2574a9',
    'knowledge:distractors': '#216a94',
    'math': '#e74c3c',
    'code': '#9b59b6',
    'loss': '#f1c40f',
}
CATEGORY_COLORS_SMALL = {cat: color for cat, color in CATEGORY_COLORS.items() if ':' not in cat}

SIZE_COLORS = {
    '4M': 'brown',
    '6M': '#7f7f7f',  # gray
    '8M': '#17becf',  # cyan
    '10M': '#bcbd22', # olive
    '14M': '#e377c2', # pink
    '16M': '#8c564b', # brown
    '20M': 'black',
    '60M': 'teal',
    '90M': 'pink',
    '150M': '#1f77b4',
    '300M': '#2ca02c',
    '530M': '#ff7f0e',
    '750M': '#d62728',
    '1B': '#9467bd'
}