from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Set pad_token_id to eos_token_id (important for attention_mask)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("The Model is ready! Type 'Quit' to exit.")
chat_history_ids = None  # to track conversation

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Exiting. Thank you for chatting!")
        break

    # Skip empty inputs
    if not user_input.strip():
        continue

    # Tokenize user input and add eos_token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Concatenate with chat history if available
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Create attention mask to avoid pad/eos confusion
    attention_mask = bot_input_ids != tokenizer.pad_token_id

    # Generate model response
    chat_history_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75
    )

    # Decode only the newly generated tokens (exclude what was input)
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    print("Bot:", response)

