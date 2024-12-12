from huggingface_hub import snapshot_download
model_id="HuggingFaceTB/SmolLM-135M"
snapshot_download(repo_id=model_id, local_dir="hf-smol",
                  local_dir_use_symlinks=False, revision="main")