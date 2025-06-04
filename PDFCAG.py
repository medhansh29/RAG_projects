from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import time


# Load dataset (allow trusted code)
dataset = load_dataset("daily_dialog", trust_remote_code=True)
dialogs = dataset["train"]

# Load GPT-2 model and tokenizer (no sentencepiece)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Add pad token to tokenizer (GPT2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(context):
    # Format prompt from context
    prompt = f"The user said: \"{context}\". Based on this, recommend a movie:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    
    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )

    # Decode and clean result
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated[len(prompt):].strip().split('\n')[0]
    return response

# 🔁 Example runs
contexts = [
    "User likes action movies",
    "User is interested in space exploration",
    "User loves classic literature"
]

for context in contexts:
    start_time = time.time()
    result = generate_response(context)
    elapsed_time = time.time() - start_time
    print(f"Context: {context}\nGenerated: {result}\n")
    print(f"⏱️ Time taken: {elapsed_time:.2f} seconds\n")
    
