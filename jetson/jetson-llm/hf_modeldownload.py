from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-8B-GGUF",
    local_dir="./models/qwen3-gguf",
    resume_download=True,
    local_dir_use_symlinks=False
)