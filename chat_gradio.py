import gradio as gr
from transformers import pipeline, AutoTokenizer

# Load model and tokenizer
model_path = "models/llama3-8b-merged-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_path)

chat = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    torch_dtype="auto",
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id
)

# Initial context
conversation_context = (
    "You are Michael Fried, a renowned art critic and historian. "
    "You respond in your own intellectual and formal voice, referencing your essays and ideas."
)

# Chat function for Gradio
def chat_with_michael(message, history):
    # Reconstruct full prompt using history + current message
    full_prompt = f"{conversation_context}\n\n"
    for user, assistant in history:
        full_prompt += f"User: {user}\nMichael Fried: {assistant}\n"
    full_prompt += f"User: {message}\nMichael Fried:"

    response = chat(
        full_prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = response[0]["generated_text"]
    generated_reply = generated_text[len(full_prompt):].strip()

    return generated_reply

# Launch interface
gr.ChatInterface(
    fn=chat_with_michael,
    title="Talk to an LLM traind on Michael Fried work",
    #description="Chat with the art historian in his signature intellectual style.",
    theme="soft",
    cache_examples=False
).launch(share=True, server_port=9128)

