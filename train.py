# preprocessing.py
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback, EvalPrediction
from peft import LoraConfig, get_peft_model
from accelerate import infer_auto_device_map
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
import torch
import sys
import os
import re
# Disable unnecessary logging and parallelism
os.environ["CLEARML_LOGGER_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.setrecursionlimit(10000)  # Increase the recursion limit

# Model name
MODEL_NAME = "meta-llama/Llama-3.1-8B"  # Replace with your model name
#MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Requires approval

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def split_into_chunks(text: str, max_tokens: int = 512, stride: int = 128) -> list:
    text = re.sub(r'\s+', ' ', text)
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
        if end >= len(tokens):
            break
        start += max_tokens - stride  # overlap
    return chunks

def preprocess_dataset(corpus_df: pd.DataFrame, max_tokens: int = 512) -> Dataset:
    """
    Preprocess the dataset by splitting long texts into chunks.
    """
    chunked_texts = []
    for text in corpus_df["text"]:
        chunks = split_into_chunks(text, max_tokens)
        chunked_texts.extend(chunks)
    dataset = Dataset.from_dict({"text": chunked_texts})

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_tokens,
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    return dataset.map(tokenize_function, batched=True, batch_size=4)

# Load your corpus
corpus_df = pd.read_parquet("data/processed/greenberg.parquet")

# Preprocess and chunk the dataset
dataset = preprocess_dataset(corpus_df)

# Split dataset into training and evaluation sets
split = dataset.train_test_split(test_size=0.10)
train_dataset, eval_dataset = split["train"], split["test"]
print(f"Dataset size: {len(dataset)}, training: {len(train_dataset)}, validation: {len(eval_dataset)}")

# Load the pre-trained model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,  # Quantization for memory efficiency
    torch_dtype=torch.bfloat16,  # Use bfloat16 for mixed precision
    device_map="auto",  # Automatically distribute model across GPUs
    offload_folder="offload",  # Offload to CPU if needed
    offload_state_dict=True,  # Offload model state dict to CPU
)

# Define LoRA configuration
peft_config = LoraConfig(
    r=8,  # Rank of the low-rank adaptation
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05,  # Dropout for LoRA layers
    bias="none",  # No bias added
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Enable gradient checkpointing to save memor
#for name, param in model.named_parameters():
 #   if "lora" in name:  # LoRA parameters should be trainable
  #      param.requires_grad = True
for name, param in model.named_parameters():
    if not any(lora_layer in name for lora_layer in ["lora"]):
        param.requires_grad = False

#model.gradient_checkpointing_enable()


from rouge_score import rouge_scorer

def compute_metrics(eval_pred):
    """
    Compute ROUGE scores using the `rouge-score` library.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Get predictions and labels
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    # Convert logits to token IDs (if predictions are logits)
    if predictions.ndim == 3:  # Shape: (batch_size, sequence_length, vocab_size)
        predictions = np.argmax(predictions, axis=-1)

    # Decode predictions and labels
    predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    # Compute ROUGE scores
    scores = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
    for pred, label in zip(predictions, labels):
        result = scorer.score(pred, label)
        scores["rouge-1"] += result["rouge1"].fmeasure
        scores["rouge-2"] += result["rouge2"].fmeasure
        scores["rouge-l"] += result["rougeL"].fmeasure

    # Average scores
    scores = {k: v / len(predictions) for k, v in scores.items()}
    return scores


# Training arguments
training_args = TrainingArguments(
    output_dir="./models/greenberg",
    per_device_train_batch_size=6,
    #gradient_accumulation_steps=2,
    learning_rate=1e-3,
    num_train_epochs=128,  # You might want to increase this if you had more epochs planned
    save_steps=512,
    logging_steps=64,
    push_to_hub=False,
    dataloader_num_workers=64,
    # Add any other training arguments you were using
)
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    #eval_dataset=eval_dataset,  # Evaluation dataset
    #compute_metrics=compute_metrics,  # Custom metric function
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("models/final_greenberg")
