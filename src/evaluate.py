import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_perplexity(model_dir, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for t in texts:
            inputs = tokenizer(t, return_tensors="pt")
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            tokens = inputs["input_ids"].numel()
            total_loss += loss * tokens
            total_tokens += tokens
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

if __name__ == "__main__":
    sample_texts = [
        "USER: I want to know about crop insurance.\nBOT: Crop insurance covers you for losses.",
    ]
    print("Perplexity:", compute_perplexity("./models/villageconnect-dialo", sample_texts))
