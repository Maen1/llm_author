from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel


BASE_MODEL= "meta-llama/Llama-3.1-8B" 
PEFT_MODEL = "models/llama_author8b/checkpoint-75776"

MERGED_MODEL = "author_merged"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Apply LoRA
model = PeftModel.from_pretrained(model, PEFT_MODEL)

# Merge LoRA into base weights
model = model.merge_and_unload()

# Save the merged model
model.save_pretrained(MERGED_MODEL)
tokenizer.save_pretrained(MERGED_MODEL)