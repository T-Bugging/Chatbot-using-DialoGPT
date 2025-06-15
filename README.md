Local Chatbot using DialoGPT

📄 Project Summary
This is a general-purpose conversational AI chatbot built using the pre-trained DialoGPT-medium model from Microsoft. The model runs locally on CPU using PyTorch and the Hugging Face Transformers library. It simulates human-like dialogue by maintaining chat history for context.

🛠️ Features

🤖 Uses DialoGPT-medium for generating natural replies. DialoGPT-medium is a 345 million parameter conversational AI model based on GPT-2, designed for generating human-like dialogue.

🧠 Maintains short-term memory using tokenized chat history

💬 Multi-turn dialogue with varied responses

🔁 Continues conversation until the user quits

🪶 Lightweight and CLI-based for ease of use on limited hardware

🚀 How It Works
Loads the pre-trained DialoGPT-medium model and tokenizer.

Tokenizes user input and maintains a running history of conversation.

Uses generate() with sampling (top_k, top_p, temperature) for diverse responses.

Decodes and prints model responses in a loop.

⚠️ Limitations
The model may produce irrelevant or nonsensical replies, especially in longer conversations.

Due to limited computing resources, a larger or fine-tuned model could not be used. This results in weaker performance, especially in context retention and coherence.

Works best for general small talk, not for domain-specific tasks.

📈 Future Improvements
Add web UI using React and FastAPI.

Replace DialoGPT with a more advanced or fine-tuned model (e.g., LLaMA, Falcon, GPT-Neo).

Enable long-term memory or file-based chat history.

Fine-tune on specific datasets to improve domain-specific performance (e.g., movie reviews, tech support).
