import pathlib

PROMPTS_PATH = (pathlib.Path(__file__).parent) / "prompts"

ALPACA_SFT_PROMPT_PATH = PROMPTS_PATH / "alpaca_sft.prompt"
QUESTION_ONLY_PROMPT_PATH = PROMPTS_PATH / "question_only.prompt"
R1_ZERO_PROMPT_PATH = PROMPTS_PATH / "r1_zero.prompt"
ZERO_SHOT_SYSTEM_PROMPT_PATH = PROMPTS_PATH / "zero_shot_system_prompt.prompt"

DATA_PATH = (pathlib.Path(__file__).parent.parent) / "data"
MODEL_PATH = (pathlib.Path(__file__).parent.parent) / "models"
