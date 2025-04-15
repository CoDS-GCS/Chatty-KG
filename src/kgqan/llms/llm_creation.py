import os
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms import VLLMOpenAI
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from vertexai import init
#from langchain_ollama import ChatOllama


# Possible Selections
# openai, -> gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o
# google -> gemini-1.5-flash
# deepseek -> deepseek-chat, deepseek-reasoner
# vllm -> llama3.1_local, qwen2.5_local, phi_local, llama3.1_instruct_local, mistral_local, codellama_local, deepseek_qwen

llm_type = "openai" # Choose From (openai, vllm, google, deepseek)
model_name = "gpt-4o"
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

model_to_token_length = {'gpt-3.5-turbo': 16000, 'llama3.1': 2048, "llama3_local": 8192, "llama3.1_local": 8192,
                         'deepseek-chat': 16000, 'gemini-1.5-flash': 16000, "qwen2.5_local": 34000, 'phi_local': 16000,
                         'codellama_local': 11000, 'mistral_local': 32000, 'deepseek_qwen': 34000, 'qwen_instruct': 19000,
                         'vicuna': 4096, 'granite8b_instruct': 120000, 'mistral_nemo': 73000}


# TODO Update to create one llm for all pipeline instead of creating a new llm object each request for better memory utlilization
def get_deep_seek_llm():
    llm = ChatDeepSeek(
        # model="deepseek-chat",
        model=model_name,
        temperature=0,
        max_retries=2,
    )
    return llm

def get_vllm_llm():
    llm = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="http://localhost:5000/v1",
        model_name=model_name,
        model_kwargs={"stop": ["```"]},
        temperature=0,
    )
    return llm

def get_openAI_llm():
    llm = ChatOpenAI(model=model_name, temperature=0)
    return llm

def get_google_llm():
    llm = ChatVertexAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        # stop=["```"],
        do_sample= True
    )
    return llm

def get_chatollama_llm():
    llm = ChatOllama(
        base_url='http://0.0.0.0:5000',
        model="llama3_local",
        temperature=0,
    )
    return llm

def get_llm(get_max_tokens=False):
    llm = None
    if llm_type == "openai":
        llm = get_openAI_llm()
    elif llm_type == "google":
        llm = get_google_llm()
    elif llm_type == "vllm":
        llm = get_vllm_llm()
    elif llm_type == "deepseek":
        llm = get_deep_seek_llm()

    if get_max_tokens:
        max_tokens = model_to_token_length.get(llm.model_name, None)
        return llm, max_tokens
    else:
        return llm
