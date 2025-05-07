import gradio as gr
from transformers import pipeline, AutoTokenizer



import gradio as gr
from transformers import pipeline, AutoTokenizer

# Load model and tokenizer
model_path = "models/llama3-8b-merged-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_path)

llm = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1,
    torch_dtype="auto",
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id
)

# Initial context
conversation_context = (
    "You are Michael Fried, a renowned art critic and historian. "
    "You respond in your own intellectual and formal voice, referencing your essays and ideas."
)


# RAG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.load_local("faiss_index/", embedding, allow_dangerous_deserialization=True).as_retriever() 



from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

llm_pipeline = HuggingFacePipeline(pipeline=llm)


custom_prompt = PromptTemplate.from_template(
    """
    You are Michael Fried, a renowned art critic and historian. 
    You respond in your own intellectual and formal voice, referencing your essays and ideas.

    {context}
    
    Question: {question}
    Helpful Answer:"""
)


rag_chain = RetrievalQA.from_chain_type(
    llm=llm_pipeline,
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": custom_prompt}
)




def chat_with_michael(message, history):
    conversation_history = ""
    for user, assistant in history:
        conversation_history += f"User: {user}\nMichael Fried: {assistant}\n"
    conversation_history += f"User: {message}"

    rag_input = {
        "query": conversation_history
    }
    rag_result = rag_chain(rag_input)
    answer = rag_result["result"].split("Helpful Answer:")[-1].strip()

    return answer



# Launch interface
gr.ChatInterface(
    fn=chat_with_michael,
    title = "Talk to an LLM trained on Michael Fried's work",
    #description="Chat with the art historian in his signature intellectual style.",
    theme= "soft",
    cache_examples = False
).launch(share=True, server_port=9127)

