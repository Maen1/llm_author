# llm_author
fine tune llm model on author's whole work

| File Name                             | Description                                                                                                                                               |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `extract_text_from_not_ocred_pdfs.py` | Extracts raw text from scanned (non-OCRed) PDFs, typically using OCR tools like Tesseract. First step in preparing data for indexing or training.         |
| `preprocessing.py`                    | Cleans and chunks extracted text into a structured format suitable for embedding, training, or retrieval.                                                 |
| `train.py`                            | Fine-tunes a language model using PEFT techniques (e.g., LoRA) on the preprocessed dataset.                                                               |
| `merge_base_peft.py`                  | Merges the fine-tuned PEFT adapter with the base model into a single, unified model checkpoint.                                                           |
| `quantize_merged_model.py`            | Quantizes the merged model (e.g., to 4-bit or 8-bit) for efficient inference in resource-constrained environments.                                        |
| `RAG_PEFT.ipynb`                      | Jupyter notebook for experimenting with Retrieval-Augmented Generation using the fine-tuned and/or quantized model. Useful for documentation and testing. |
| `author_style.py`                     | Defines persona-specific prompt templates (e.g., simulating Michael Fried's style) for consistent model responses.                                        |
| `RAG_chat_gradio.py`                  | Gradio-based chatbot that integrates a retriever with your LLM to provide context-aware, author-styled answers.                                           |
| `chat_gradio.py`                      | A simpler Gradio chatbot interface using only the language model without retrieval-based augmentation.                                                    |

