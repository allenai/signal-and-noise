RC_TASKS_MMLU = [
    "mmlu_abstract_algebra:rc::olmes:full",
    "mmlu_anatomy:rc::olmes:full",
    "mmlu_astronomy:rc::olmes:full",
    "mmlu_business_ethics:rc::olmes:full",
    "mmlu_clinical_knowledge:rc::olmes:full",
    "mmlu_college_biology:rc::olmes:full",
    "mmlu_college_chemistry:rc::olmes:full",
    "mmlu_college_computer_science:rc::olmes:full",
    "mmlu_college_mathematics:rc::olmes:full",
    "mmlu_college_medicine:rc::olmes:full",
    "mmlu_college_physics:rc::olmes:full",
    "mmlu_computer_security:rc::olmes:full",
    "mmlu_conceptual_physics:rc::olmes:full",
    "mmlu_econometrics:rc::olmes:full",
    "mmlu_electrical_engineering:rc::olmes:full",
    "mmlu_elementary_mathematics:rc::olmes:full",
    "mmlu_formal_logic:rc::olmes:full",
    "mmlu_global_facts:rc::olmes:full",
    "mmlu_high_school_biology:rc::olmes:full",
    "mmlu_high_school_chemistry:rc::olmes:full",
    "mmlu_high_school_computer_science:rc::olmes:full",
    "mmlu_high_school_european_history:rc::olmes:full",
    "mmlu_high_school_geography:rc::olmes:full",
    "mmlu_high_school_government_and_politics:rc::olmes:full",
    "mmlu_high_school_macroeconomics:rc::olmes:full",
    "mmlu_high_school_mathematics:rc::olmes:full",
    "mmlu_high_school_microeconomics:rc::olmes:full",
    "mmlu_high_school_physics:rc::olmes:full",
    "mmlu_high_school_psychology:rc::olmes:full",
    "mmlu_high_school_statistics:rc::olmes:full",
    "mmlu_high_school_us_history:rc::olmes:full",
    "mmlu_high_school_world_history:rc::olmes:full",
    "mmlu_human_aging:rc::olmes:full",
    "mmlu_human_sexuality:rc::olmes:full",
    "mmlu_international_law:rc::olmes:full",
    "mmlu_jurisprudence:rc::olmes:full",
    "mmlu_logical_fallacies:rc::olmes:full",
    "mmlu_machine_learning:rc::olmes:full",
    "mmlu_management:rc::olmes:full",
    "mmlu_marketing:rc::olmes:full",
    "mmlu_medical_genetics:rc::olmes:full",
    "mmlu_miscellaneous:rc::olmes:full",
    "mmlu_moral_disputes:rc::olmes:full",
    "mmlu_moral_scenarios:rc::olmes:full",
    "mmlu_nutrition:rc::olmes:full",
    "mmlu_philosophy:rc::olmes:full",
    "mmlu_prehistory:rc::olmes:full",
    "mmlu_professional_accounting:rc::olmes:full",
    "mmlu_professional_law:rc::olmes:full",
    "mmlu_professional_medicine:rc::olmes:full",
    "mmlu_professional_psychology:rc::olmes:full",
    "mmlu_public_relations:rc::olmes:full",
    "mmlu_security_studies:rc::olmes:full",
    "mmlu_sociology:rc::olmes:full",
    "mmlu_us_foreign_policy:rc::olmes:full",
    "mmlu_virology:rc::olmes:full",
    "mmlu_world_religions:rc::olmes:full",
]

RC_TASKS_OLMES = [
    "arc_challenge:rc::olmes:full",
    "arc_easy:rc::olmes:full",
    "boolq:rc::olmes:full",
    "csqa:rc::olmes:full",
    "hellaswag:rc::olmes:full",
    "openbookqa:rc::olmes:full",
    "piqa:rc::olmes:full",
    "socialiqa:rc::olmes:full",
    "winogrande:rc::olmes:full",
] + RC_TASKS_MMLU

MC_TASKS_OLMES = [task.replace(":rc", ":mc") for task in RC_TASKS_OLMES]

MC_TASKS_COPY_COLORS = [
    "copycolors_4way:mc::none",
]

PALOMA = [
    # "paloma_c4_100_domains::paloma", # 28K
    # "paloma_dolma_100_subreddits::paloma", # 92K
    # "paloma_twitterAAE_HELM_fixed::paloma", # 100K
    "paloma_4chan_meta_sep::paloma",
    "paloma_c4_en::paloma",
    "paloma_dolma_100_programing_languages::paloma",
    "paloma_dolma-v1_5::paloma",
    "paloma_falcon-refinedweb::paloma",
    "paloma_gab::paloma",
    "paloma_m2d2_s2orc_unsplit::paloma",
    "paloma_m2d2_wikipedia_unsplit::paloma",
    "paloma_manosphere_meta_sep::paloma",
    "paloma_mc4::paloma",
    "paloma_ptb::paloma",
    "paloma_redpajama::paloma",
    "paloma_wikitext_103::paloma",
]
LLM_COMPRESSION = [
    "arxiv_math::llm_compression",
    "cc::llm_compression",
    "python::llm_compression",
]
CUSTOM_LOSS = [
    'sky_t1::custom_loss', 
    'numia_math::custom_loss', 
    'tulu_if::custom_loss'
]

GEN_TASKS_OLMES = [
    # "coqa::olmes:full", # <- coqa is not setup properly (no few shot examples)
    # "gsm8k::olmes:full", # <- already included elsewhere under a different name
    "drop::olmes:full",
    "jeopardy::olmes:full",
    "naturalqs::olmes:full",
    "squad::olmes:full",
    "triviaqa::olmes:full",
]

MMLU_PRO_MC = [
    "mmlu_pro_math:mc::none",
    "mmlu_pro_health:mc::none",
    "mmlu_pro_physics:mc::none",
    "mmlu_pro_business:mc::none",
    "mmlu_pro_biology:mc::none",
    "mmlu_pro_chemistry:mc::none",
    "mmlu_pro_computer science:mc::none",
    "mmlu_pro_economics:mc::none",
    "mmlu_pro_engineering:mc::none",
    "mmlu_pro_philosophy:mc::none",
    "mmlu_pro_other:mc::none",
    "mmlu_pro_history:mc::none",
    "mmlu_pro_psychology:mc::none",
    "mmlu_pro_law:mc::none",
]
MMLU_PRO_RC  = [task.replace(":mc::none", ":rc::none") for task in MMLU_PRO_MC]
MMLU_PRO_COT = [task.replace(":mc::none", ":cot::none") for task in MMLU_PRO_MC]

AGI_EVAL_MC = [
    "agi_eval_lsat-ar::olmes:full",
    "agi_eval_lsat-lr::olmes:full",
    "agi_eval_lsat-rc::olmes:full",
    "agi_eval_logiqa-en::olmes:full",
    "agi_eval_sat-math::olmes:full",
    "agi_eval_sat-en::olmes:full",
    "agi_eval_aqua-rat::olmes:full",
    "agi_eval_sat-en-without-passage::olmes:full",
    "agi_eval_gaokao-english::olmes:full",
]
AGI_EVAL_RC  = [task.replace("::olmes:full", ":rc::none") for task in AGI_EVAL_MC]
AGI_EVAL_COT = [task.replace("::olmes:full", ":cot::none") for task in AGI_EVAL_MC]

MINERVA_COT = [
    "minerva_math_algebra::olmes:full",
    "minerva_math_counting_and_probability::olmes:full",
    "minerva_math_geometry::olmes:full",
    "minerva_math_intermediate_algebra::olmes:full",
    "minerva_math_number_theory::olmes:full",
    "minerva_math_prealgebra::olmes:full",
    "minerva_math_precalculus::olmes:full",
]

BBH_COT = [
    "bbh_boolean_expressions:cot::olmes:full",
    "bbh_causal_judgement:cot::olmes:full",
    "bbh_date_understanding:cot::olmes:full",
    "bbh_disambiguation_qa:cot::olmes:full",
    "bbh_dyck_languages:cot::olmes:full",
    "bbh_formal_fallacies:cot::olmes:full",
    "bbh_geometric_shapes:cot::olmes:full",
    "bbh_hyperbaton:cot::olmes:full",
    "bbh_logical_deduction_five_objects:cot::olmes:full",
    "bbh_logical_deduction_seven_objects:cot::olmes:full",
    "bbh_logical_deduction_three_objects:cot::olmes:full",
    "bbh_movie_recommendation:cot::olmes:full",
    "bbh_multistep_arithmetic_two:cot::olmes:full",
    "bbh_navigate:cot::olmes:full",
    "bbh_object_counting:cot::olmes:full",
    "bbh_penguins_in_a_table:cot::olmes:full",
    "bbh_reasoning_about_colored_objects:cot::olmes:full",
    "bbh_ruin_names:cot::olmes:full",
    "bbh_salient_translation_error_detection:cot::olmes:full",
    "bbh_snarks:cot::olmes:full",
    "bbh_sports_understanding:cot::olmes:full",
    "bbh_temporal_sequences:cot::olmes:full",
    "bbh_tracking_shuffled_objects_five_objects:cot::olmes:full",
    "bbh_tracking_shuffled_objects_seven_objects:cot::olmes:full",
    "bbh_tracking_shuffled_objects_three_objects:cot::olmes:full",
    "bbh_web_of_lies:cot::olmes:full",
    "bbh_word_sorting:cot::olmes:full",
]

AUTOBENCHER = [
    'autobencher::none', 
    'autobencher:mc::none'
]

MATH_CODE = [
    "gsm8k::olmes:full",
    "minerva_math_algebra::olmes:full",
    "minerva_math_counting_and_probability::olmes:full",
    "minerva_math_geometry::olmes:full",
    "minerva_math_intermediate_algebra::olmes:full",
    "minerva_math_number_theory::olmes:full",
    "minerva_math_prealgebra::olmes:full",
    "minerva_math_precalculus::olmes:full",
    "mbpp::ladder",
    "mbppplus::ladder",
    "codex_humaneval:temp0.8",
    "codex_humanevalplus::ladder", 
    'gsm_plus::none',
    'gsm_symbolic::none',
    'gsm_symbolic_p1::none',
    'gsm_symbolic_p2::none',
    'minerva_math_500::none', 
]

EXTA_TASKS = [
    # 'gpqa::none', # requires HF token
    'deepmind_math_large::none',
    'medmcqa:rc::none',
    'medmcqa:mc::none',
    'aime::none',
]