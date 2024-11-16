import shutil
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def save_model_details(local_save_path, model_alias, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tok_path = os.path.join(local_save_path, model_alias + "_tokenizer")
    shutil.rmtree(tok_path, ignore_errors=True)
    tokenizer.save_pretrained(tok_path)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    mod_path = os.path.join(local_save_path, model_alias + "_model")
    shutil.rmtree(mod_path, ignore_errors=True)
    model.save_pretrained(mod_path)

    print(model)
    print(f"saved {model_alias} model and tokenizer to {local_save_path}")


local_save_path = "models"
save_model_details(local_save_path, "gpt-neo", "EleutherAI/gpt-neo-125m")
save_model_details(local_save_path, "fb-opt", "facebook/opt-125m")
