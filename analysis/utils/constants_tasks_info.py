PRETTY_BENCHMARK_NAMES = {
    'mmlu': 'MMLU',
    'squad': 'SQuAD',
    'gsm8k': 'GSM8K',
    'hellaswag': 'HellaSwag',
    'arc_easy': 'ARC Easy',
    'arc_challenge': 'ARC Challenge',
    'david_task_easy': 'Paraphrased ARC Easy',
    'david_task_challenge': 'Paraphrased ARC Challenge',
    'boolq': 'BoolQ',
    'csqa': 'CommonsenseQA',
    'openbookqa': 'OpenBookQA',
    'piqa': 'PIQA',
    'socialiqa': 'SocialIQA',
    'winogrande': 'WinoGrande',

    'mmlu_stem': 'MMLU STEM',
    'mmlu_humanities': 'MMLU Humanities',
    'mmlu_social_sciences': 'MMLU Social Sciences',
    'mmlu_other': 'MMLU Other',
}

RANDOM_CHANCE_OLMES = {
    "mmlu": 0.25,
    "squad": 0.50,
    "openbookqa": 0.25,
    "socialiqa": 1 / 3,
    "winogrande": 0.50,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "piqa": 0.5,
    "boolq": 0.5,
    "hellaswag": 0.25,
    "copa": 0.5,
    "csqa": 0.2,
}

RANDOM_CHANCE_SAMIR = {
    "AGIEval LSAT AR": 0.25,
    "AGIEval LSAT LR": 0.25,
    "AGIEval LSAT RC": 0.25,
    "AGIEval SAT English": 0.25,
    "ARC-Challenge": 0.25,
    "ARC-Easy": 0.25,
    "BBQ": 0.25,
    "BIG-bench: CS algorithms": 0.50,
    "BIG-bench: Conceptual combinations": 0.00,
    "BIG-bench: Conlang translation": 0.25,
    "BIG-bench: Dyck languages": 0.00,
    "BIG-bench: Elementary math QA": 0.00,
    "BIG-bench: Language identification": 0.25,
    "BIG-bench: Logical deduction": 0.25,
    "BIG-bench: Misconceptions": 0.25,
    "BIG-bench: Novel Concepts": 0.50,
    "BIG-bench: Operators": 0.25,
    "BIG-bench: QA WikiData": 0.00,
    "BIG-bench: Repeat copy logic": 0.00,
    "BIG-bench: Strange stories": 0.00,
    "BIG-bench: Strategy QA": 0.50,
    "BIG-bench: Understanding fables": 0.50,
    "BoolQ": 0.25,
    "COPA": 0.50,
    "CoQA": 0.50,
    "Commonsense QA": 0.00,
    "Enterprise PII classification": 0.25,
    "HellaSwag (10-shot)": 0.50,
    "HellaSwag (zero-shot)": 0.25,
    "Jeopardy": 0.25,
    "LAMBADA": 0.00,
    "LogiQA": 0.00,
    "MMLU (5-shot)": 0.25,
    "MMLU (zero-shot)": 0.25,
    "MathQA": 0.25,
    "OpenBook QA": 0.25,
    "PIQA": 0.25,
    "PubMed QA Labeled": 0.50,
    "SIQA": 0.00,
    "SQuAD": 0.50,
    "Simple Arithmetic: NoSpaces": 0.00,
    "Simple Arithmetic: WithSpaces": 0.00,
    "WinoGender MC: Female": 0.00,
    "WinoGender MC: Male": 0.50,
    "WinoGrande": 0.50,
    "WinoGrand": 0.50
}

ALL_MMLU_SUBSETS = ['mmlu_abstract_algebra', 'mmlu_anatomy', 'mmlu_astronomy', 'mmlu_business_ethics', 'mmlu_clinical_knowledge', 'mmlu_college_biology', 'mmlu_college_chemistry', 'mmlu_college_computer_science', 'mmlu_college_mathematics', 'mmlu_college_medicine', 'mmlu_college_physics', 'mmlu_computer_security', 'mmlu_conceptual_physics', 'mmlu_econometrics', 'mmlu_electrical_engineering', 'mmlu_elementary_mathematics', 'mmlu_formal_logic', 'mmlu_global_facts', 'mmlu_high_school_biology', 'mmlu_high_school_chemistry', 'mmlu_high_school_computer_science', 'mmlu_high_school_european_history', 'mmlu_high_school_geography', 'mmlu_high_school_government_and_politics', 'mmlu_high_school_macroeconomics', 'mmlu_high_school_mathematics', 'mmlu_high_school_microeconomics', 'mmlu_high_school_physics', 'mmlu_high_school_psychology', 'mmlu_high_school_statistics', 'mmlu_high_school_us_history', 'mmlu_high_school_world_history', 'mmlu_human_aging', 'mmlu_human_sexuality', 'mmlu_international_law', 'mmlu_jurisprudence', 'mmlu_logical_fallacies', 'mmlu_machine_learning', 'mmlu_management', 'mmlu_marketing', 'mmlu_medical_genetics', 'mmlu_miscellaneous', 'mmlu_moral_disputes', 'mmlu_moral_scenarios', 'mmlu_nutrition', 'mmlu_philosophy', 'mmlu_prehistory', 'mmlu_professional_accounting', 'mmlu_professional_law', 'mmlu_professional_medicine', 'mmlu_professional_psychology', 'mmlu_public_relations', 'mmlu_security_studies', 'mmlu_sociology', 'mmlu_us_foreign_policy', 'mmlu_virology', 'mmlu_world_religions']

# https://github.com/hendrycks/test/blob/master/categories.py
MMLU_SUBCATEGORIES = {
    "abstract_algebra": ["math"], "anatomy": ["health"], "astronomy": ["physics"], "business_ethics": ["business"], "clinical_knowledge": ["health"], "college_biology": ["biology"], "college_chemistry": ["chemistry"], "college_computer_science": ["computer science"], "college_mathematics": ["math"], "college_medicine": ["health"], "college_physics": ["physics"], "computer_security": ["computer science"], "conceptual_physics": ["physics"], "econometrics": ["economics"], "electrical_engineering": ["engineering"], "elementary_mathematics": ["math"], "formal_logic": ["philosophy"], "global_facts": ["other"], "high_school_biology": ["biology"], "high_school_chemistry": ["chemistry"], "high_school_computer_science": ["computer science"], "high_school_european_history": ["history"], "high_school_geography": ["geography"], "high_school_government_and_politics": ["politics"], "high_school_macroeconomics": ["economics"], "high_school_mathematics": ["math"], "high_school_microeconomics": ["economics"], "high_school_physics": ["physics"], "high_school_psychology": ["psychology"], "high_school_statistics": ["math"], "high_school_us_history": ["history"], "high_school_world_history": ["history"], "human_aging": ["health"], "human_sexuality": ["culture"], "international_law": ["law"], "jurisprudence": ["law"], "logical_fallacies": ["philosophy"], "machine_learning": ["computer science"], "management": ["business"], "marketing": ["business"], "medical_genetics": ["health"], "miscellaneous": ["other"], "moral_disputes": ["philosophy"], "moral_scenarios": ["philosophy"], "nutrition": ["health"], "philosophy": ["philosophy"], "prehistory": ["history"], "professional_accounting": ["other"], "professional_law": ["law"], "professional_medicine": ["health"], "professional_psychology": ["psychology"], "public_relations": ["politics"], "security_studies": ["politics"], "sociology": ["culture"], "us_foreign_policy": ["politics"], "virology": ["health"], "world_religions": ["philosophy"],
}

MMLU_CATEGORIES = {
    "stem": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social_sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other": ["other", "business", "health"],
}