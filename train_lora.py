# train_lora.py
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# 1. Model Configuration
model_id = "Qwen/Qwen2.5-7B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"": 0},
    trust_remote_code=True
)

# 2. Prepare Model for QLoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Optimized for Qwen architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Displays % of parameters being trained (~0.1%)

# 3. Load Dataset
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# 4. Training Arguments (Tailored for 8GB GPU)
training_args = SFTConfig(
    output_dir="./qwen25-ctf-lora-results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    dataset_text_field="messages",  # TRL automatically formats Qwen messages
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

# 5. Execute Training
print("Starting QLoRA Fine-Tuning for Qwen 2.5 7B...")
trainer.train()

# 6. Save Adapter Weights
adapter_path = "qwen25-ctf-lora-adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"Fine-tuning complete! Adapter saved to ./{adapter_path}")