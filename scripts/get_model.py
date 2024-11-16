import shutil
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

gpt_neo_model_name = "EleutherAI/gpt-neo-125m"
gpt_neo_tokenizer = AutoTokenizer.from_pretrained(gpt_neo_model_name)
gpt_neo_model = AutoModelForCausalLM.from_pretrained(gpt_neo_model_name)

fb_opt_model_name = "facebook/opt-125m"
fb_opt_tokenizer = AutoTokenizer.from_pretrained(fb_opt_model_name)
fb_opt_model = AutoModelForCausalLM.from_pretrained(fb_opt_model_name)


def save_models(model, tokenizer, local_save_path, model_alias):
    # save the model and tokenizer
    tok_path = os.path.join(local_save_path, model_alias + "_tokenizer")
    shutil.rmtree(tok_path, ignore_errors=True)
    tokenizer.save_pretrained(os.path.join(local_save_path, model_alias + "_tokenizer"))

    mod_path = os.path.join(local_save_path, model_alias + "_model")
    shutil.rmtree(mod_path, ignore_errors=True)
    model.save_pretrained(os.path.join(local_save_path, model_alias + "_model"))

    print(f"saved {model_alias} model and tokenizer to {local_save_path}")


local_save_path = "models"
save_models(gpt_neo_model, gpt_neo_tokenizer, local_save_path, "gpt-neo")
save_models(fb_opt_model, fb_opt_tokenizer, local_save_path, "fb-opt")
