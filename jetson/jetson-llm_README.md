#🎯 Ollama on Jetson with GPU Support

Ollama is a popular open-source tool that allows users to easily run a large language models (LLMs) locally on their own computer, serving as an accessible entry point to LLMs for many.

It now offers out-of-the-box support for the Jetson platform with CUDA support, enabling Jetson users to seamlessly install Ollama with a single command and start using it immediately.

Ollama uses llama.cpp for inference, which various API benchmarks and comparisons are provided for on the Llava page. It gets roughly half of peak performance versus the faster APIs like NanoLLM , but is generally considered fast enough for text chat.

Supported hardware: Orin Nano (8GB), Orin NX (16GB), AGX Orin (32/64GB)
Supported JetPack versions: 5.x (L4T r35.x) and 6.x (L4T r36.x)

🧩 Installation Methods

✅ 1. Native Install (Recommended)
```bash
sjsujetson@sjsujetson-01:~/Developer$ curl -fsSL https://ollama.com/install.sh | sh
>>> Installing ollama to /usr/local
[sudo] password for sjsujetson: 
>>> Downloading Linux arm64 bundle
######################################################################## 100.0%
>>> Downloading JetPack 6 components
######################################################################## 100.0%
>>> Creating ollama user...
>>> Adding ollama user to render group...
>>> Adding ollama user to video group...
>>> Adding current user to ollama group...
>>> Creating ollama systemd service...
>>> Enabling and starting ollama service...
Created symlink /etc/systemd/system/default.target.wants/ollama.service → /etc/systemd/system/ollama.service.
>>> NVIDIA JetPack ready.
>>> The Ollama API is now available at 127.0.0.1:11434.
>>> Install complete. Run "ollama" from the command line.
```

Run llama model:
```bash
sjsujetson@sjsujetson-01:~/Developer$ ollama run llama3.2:3b
```
End a chat session, just type: `/exit`
Reset the chat context (clear memory): `/reset`

Pull a model (download from Ollama Hub):
```bash
ollama pull phi3
ollama pull deepseek-coder
sjsujetson@sjsujetson-01:~/Developer$ ollama pull qwen2
ollama run qwen2
sjsujetson@sjsujetson-01:~/Developer$ ollama show qwen2
  Model
    architecture        qwen2    
    parameters          7.6B     
    context length      32768    
    embedding length    3584     
    quantization        Q4_0
```

List all available local models:
```bash
ollama list
```

Docker image:
```bash
sudo docker build --network=host -t jetson-llm .
```

# llama.cpp on Jetson with GPU Support
[llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

Download models:
```bash
sjsujetson@sjsujetson-01:~/Developer/models$ wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O mistral.gguf
```

Enter the docker container:
```bash
sjsujetson@sjsujetson-01:~/Developer/models$ docker run -it --rm \
  --runtime nvidia \
  --network host \
  -v ~/Developer:/workspace \
  jetson-pytorch-v1
```

Perform the basic llama.cpp build for CPU:
```bash
root@sjsujetson-01:/workspace# git clone https://github.com/ggerganov/llama.cpp.git
root@sjsujetson-01:/workspace/llama.cpp# cmake -B build
.....
-- ARM feature DOTPROD enabled
-- ARM feature FMA enabled
-- ARM feature FP16_VECTOR_ARITHMETIC enabled
-- Adding CPU backend variant ggml-cpu: -mcpu=cortex-a78ae+crc+crypto+flagm+pauth+nossbs+dotprod+noi8mm+nosve 
-- Found CURL: /usr/lib/aarch64-linux-gnu/libcurl.so (found version "8.5.0")
-- Configuring done (6.2s)
-- Generating done (0.3s)
-- Build files have been written to: /workspace/llama.cpp/build
root@sjsujetson-01:/workspace/llama.cpp# cmake --build build --config Release
```


Create a new build directory for CUDA support:
```bash
root@sjsujetson-01:/workspace/llama.cpp# mkdir build_cuda
root@sjsujetson-01:/workspace/llama.cpp# cmake -B build_cuda -DGGML_CUDA=ON
....
-- ARM feature DOTPROD enabled
-- ARM feature FMA enabled
-- ARM feature FP16_VECTOR_ARITHMETIC enabled
-- Adding CPU backend variant ggml-cpu: -mcpu=cortex-a78ae+crc+crypto+flagm+pauth+nossbs+dotprod+noi8mm+nosve 
-- Found CUDAToolkit: /usr/local/cuda/targets/aarch64-linux/include (found version "12.6.85")
-- CUDA Toolkit found
-- Using CUDA architectures: native
-- The CUDA compiler identification is NVIDIA 12.6.85 with host compiler GNU 13.2.0
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- CUDA host compiler is GNU 13.2.0
-- Including CUDA backend
-- Found CURL: /usr/lib/aarch64-linux-gnu/libcurl.so (found version "8.5.0")
-- Configuring done (22.5s)
-- Generating done (0.4s)
-- Build files have been written to: /workspace/llama.cpp/build_cuda
```

```bash
root@sjsujetson-01:/workspace/llama.cpp# cmake --build build_cuda --config Release
....
[100%] Built target llama-export-lora
root@sjsujetson-01:/workspace/llama.cpp# ls build_cuda
CMakeCache.txt       DartConfiguration.tcl  bin                  compile_commands.json  llama-config.cmake   pocs   tools
CMakeFiles           Makefile               cmake_install.cmake  examples               llama-version.cmake  src
CTestTestfile.cmake  Testing                common               ggml                   llama.pc             tests
root@sjsujetson-01:/workspace/llama.cpp# ls build_cuda/bin/ #contains all the executable files
```

llama.cpp requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format.

`llama-cli` is a CLI tool for accessing and experimenting with most of llama.cpp's functionality. Run in conversation mode: `llama-cli -m model.gguf` or add custom chat template: `llama-cli -m model.gguf -cnv --chat-template chatml`

Run a local downloaded model:
```bash
root@sjsujetson-01:/workspace/llama.cpp# ./build_cuda/bin/llama-cli -m ../models/mistral.gguf -p "Explain what is Nvidia jetson"
....
llama_perf_sampler_print:    sampling time =      34.98 ms /   532 runs   (    0.07 ms per token, 15210.86 tokens per second)
llama_perf_context_print:        load time =    3498.72 ms
llama_perf_context_print: prompt eval time =    2193.93 ms /    17 tokens (  129.05 ms per token,     7.75 tokens per second)
llama_perf_context_print:        eval time =   84805.65 ms /   514 runs   (  164.99 ms per token,     6.06 tokens per second)
llama_perf_context_print:       total time =   92930.78 ms /   531 tokens
```

`llama-server` is a lightweight, OpenAI API compatible, HTTP server for serving LLMs. Start a local HTTP server with default configuration on port 8080: `llama-server -m model.gguf --port 8080`, Basic web UI can be accessed via browser: `http://localhost:8080`. Chat completion endpoint: `http://localhost:8080/v1/chat/completions`
```bash
root@sjsujetson-01:/workspace/llama.cpp# ./build_cuda/bin/llama-server -m ../models/mistral.gguf --port 8080
....
ggml_cuda_init: found 1 CUDA devices:
  Device 0: Orin, compute capability 8.7, VMM: yes
....
main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 5
main: loading model
srv    load_model: loading model '../models/mistral.gguf'
llama_model_load_from_file_impl: using device CUDA0 (Orin) - 5818 MiB free
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle
slot launch_slot_: id  0 | task 0 | processing task
slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 4096, n_keep = 0, n_prompt_tokens = 11
slot update_slots: id  0 | task 0 | kv cache rm [0, end)
slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 11, n_tokens = 11, progress = 1.000000
slot update_slots: id  0 | task 0 | prompt done, n_past = 11, n_tokens = 11
slot      release: id  0 | task 0 | stop processing: n_past = 110, truncated = 0
slot print_timing: id  0 | task 0 | 
prompt eval time =    1411.71 ms /    11 tokens (  128.34 ms per token,     7.79 tokens per second)
       eval time =   16087.67 ms /   100 tokens (  160.88 ms per token,     6.22 tokens per second)
      total time =   17499.37 ms /   111 tokens
srv  update_slots: all slots are idle
srv  log_server_r: request: POST /completion 127.0.0.1 200
```

```bash
Send request via curl in another terminal (in the host machine or container):
sjsujetson@sjsujetson-01:~$ curl http://localhost:8080/completion -d '{
  "prompt": "Explain what is Nvidia jetson?",
  "n_predict": 100
}'

{"index":0,"content":"\n\nNvidia Jetson is a series of embedded computing platforms developed by Nvidia for artificial intelligence (AI), deep learning, and autonomous machines. These platforms are designed to provide the computational power and capabilities of a data center in an embedded form factor, making them suitable for edge computing applications.\n\nJetson devices include GPU processors, CPU cores, and specialized hardware accelerators for deep learning inference, computer vision, and video processing. They also offer a","tokens":[],"id_slot":0,"stop":true,"model":"gpt-3.5-turbo","tokens_predicted":100,"tokens_evaluated":11,"generation_settings":{"n_predict":100,"seed":4294967295,"temperature":0.800000011920929,"dynatemp_range":0.0,"dynatemp_exponent":1.0,"top_k":40,"top_p":0.949999988079071,"min_p":0.05000000074505806,"top_n_sigma":-1.0,"xtc_probability":0.0,"xtc_threshold":0.10000000149011612,"typical_p":1.0,"repeat_last_n":64,"repeat_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0,"dry_multiplier":0.0,"dry_base":1.75,"dry_allowed_length":2,"dry_penalty_last_n":4096,"dry_sequence_breakers":["\n",":","\"","*"],"mirostat":0,"mirostat_tau":5.0,"mirostat_eta":0.10000000149011612,"stop":[],"max_tokens":100,"n_keep":0,"n_discard":0,"ignore_eos":false,"stream":false,"logit_bias":[],"n_probs":0,"min_keep":0,"grammar":"","grammar_lazy":false,"grammar_triggers":[],"preserved_tokens":[],"chat_format":"Content-only","reasoning_format":"deepseek","reasoning_in_content":false,"thinking_forced_open":false,"samplers":["penalties","dry","top_n_sigma","top_k","typ_p","top_p","min_p","xtc","temperature"],"speculative.n_max":16,"speculative.n_min":0,"speculative.p_min":0.75,"timings_per_token":false,"post_sampling_probs":false,"lora":[]},"prompt":"<s> Explain what is Nvidia jetson?","has_new_line":true,"truncated":false,"stop_type":"limit","stopping_word":"","tokens_cached":110,"timings":{"prompt_n":11,"prompt_ms":1411.706,"prompt_per_token_ms":128.33690909090907,"prompt_per_second":7.791990683612594,"predicted_n":100,"predicted_ms":16087.666,"predicted_per_token_ms":160.87666,"predicted_per_second":6.215942076370805}}
```

By default, llama-server listens only on 127.0.0.1 (localhost), which blocks external access. To enable external access, you need to bind to 0.0.0.0 (This tells it to accept connections from any IP address.):
```bash
./build_cuda/bin/llama-server -m ../models/mistral.gguf --port 8080 --host 0.0.0.0
```
If your Jetson device has ufw (Uncomplicated Firewall) or iptables enabled, open port 8080:
```bash
sudo ufw allow 8080/tcp
```

Make a Remote Request from Another Machine by changing the local host to the Jetson IP.

Commit changes of the container:
```bash
sjsujetson@sjsujetson-01:~$ docker ps
CONTAINER ID   IMAGE               COMMAND                  CREATED        STATUS        PORTS     NAMES
8d1a394f3fb5   jetson-pytorch-v1   "/opt/nvidia/nvidia_…"   24 hours ago   Up 24 hours             adoring_allen
sjsujetson@sjsujetson-01:~$ docker commit 8d1a394f3fb5 jetson-llm-v1
sha256:34c72d6a4a119d873ea9b56449c81be724ab377d51d6ffc776c634fec335d9e5
```
If running in Docker, also expose the port with -p:
```bash
docker run --rm -it --runtime nvidia \
  -v $(pwd)/models:/app/models \
  -p 8080:8080 \
  jetson-llama-cpp \
  ./bin/server -m /app/models/mistral.gguf --host 0.0.0.0 --port 8080
```