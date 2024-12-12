from huggingface_hub import HfApi

# Initialize the Hugging Face API with your token (optional if $HUGGING_FACE_HUB_TOKEN is set)
api = HfApi()

# Define a new repository name under your namespace
username = api.whoami()["name"]  # Get your Hugging Face username
model_id = f"{username}/SmolLM-135M"  # Change "SmolLM-135M" to your preferred model name

# Create a new repository under your namespace
api.create_repo(repo_id=model_id, exist_ok=True, repo_type="model")

# Upload the model file
api.upload_file(
    path_or_fileobj="hf-smol.gguf",
    path_in_repo="hf-smol.gguf",
    repo_id=model_id,
)

print(f"Model uploaded successfully to: https://huggingface.co/{model_id}")