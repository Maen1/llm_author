from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

MODEL_PATH = "models/llama8b-merged"  # your merged model
QUANTIZED_PATH = "model/quatized"

# Setup quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16", 
    bnb_4bit_quant_type="nf4",   
)

# Load model with quantization
MODEL_PATH = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)


# Save quantized model
model.save_pretrained("models/llama3-8b-merged-4bit")
tokenizer.save_pretrained("models/llama3-8b-merged-4bit")