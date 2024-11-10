import shutil
import os

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model_name = "EleutherAI/gpt-neo-125m"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

local_save_path = "models"
model_alias = "gpt-neo"

# delete local save path
shutil.rmtree(local_save_path, ignore_errors=True)

# save the model and tokenizer
tokenizer.save_pretrained(os.path.join(local_save_path, model_alias + "_tokenizer"))
model.save_pretrained(os.path.join(local_save_path, model_alias + "_model"))
