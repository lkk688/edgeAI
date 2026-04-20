from openai import OpenAI

client = OpenAI(
    base_url="http://100.110.236.127:8080/v1", 
    api_key="sk-no-key-required" 
)

response = client.chat.completions.create(
    model="qwen", # llama-server do not check name
    messages=[
        {"role": "system", "content": "You are a helpful Coding Assistant1"},
        {"role": "user", "content": "Please write a python code for text classification via word embedding and SVM"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)