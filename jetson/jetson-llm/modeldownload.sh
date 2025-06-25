mkdir -p models && cd models

wget -c https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O mistral.gguf

wget -c https://huggingface.co/TheBloke/Qwen1.5-7B-Chat-GGUF/resolve/main/qwen1_5-7b-chat.Q4_K_M.gguf -O qwen.gguf

wget -c https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf -O llama3.gguf

wget -c https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf -O deepseek.gguf