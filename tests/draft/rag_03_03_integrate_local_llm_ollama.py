from langchain_ollama import OllamaLLM

# 初始化本地自定义模型
model_name =  'qwen3:1.7b'
llm = OllamaLLM(
    model=model_name,  # 对应Ollama创建的自定义模型名
    temperature=0.3,
    num_ctx=32768
)

# 调用模型（集成到RAG-Agent流程）
prompt = "make a short introduction to large language model"
response = llm.invoke(prompt)
print("reply: ", response)
