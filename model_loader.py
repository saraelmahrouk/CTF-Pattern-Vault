import streamlit as st
import torch
import os
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from config import HF_TOKEN
from langchain_groq import ChatGroq
import gc
import transformers.modeling_utils

# Bypass Hugging Face's dummy VRAM warmup allocation
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

# Prevent PyTorch memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Inside get_chat_model(), right before loading AutoModelForCausalLM:
gc.collect()
torch.cuda.empty_cache()

ADAPTER_PATH = "./qwen25-ctf-lora-adapter"

@st.cache_resource(show_spinner=False)
def get_chat_model(use_local: bool = False):

    if not use_local:
        print("⚡ Using Groq API (Llama 3.3 70B)...")
        return ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.1, 
            max_retries=5
        )
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model_id = "Qwen/Qwen2.5-7B-Instruct"


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        token=HF_TOKEN
    )
    if os.path.exists(ADAPTER_PATH):
        print("Found fine-tuned adapter! Loading Qwen 2.5 + LoRA...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print("No adapter found. Loading base Qwen 2.5 model...")
        model = base_model

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        return_full_text=False,
        repetition_penalty=1.15,
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return ChatHuggingFace(llm=llm)



