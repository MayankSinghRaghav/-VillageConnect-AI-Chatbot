import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = os.environ.get("MODEL_DIR", "./models/villageconnect-dialo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VillageBot:
    def __init__(self, model_dir=MODEL_DIR):
        print("Loading model from", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(DEVICE)
        self.chat_history_ids = None

    def reset(self):
        self.chat_history_ids = None

    def chat(self, user_input, max_length=150):
        prompt = f"USER: {user_input}\nBOT:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[-1] + max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1,
            )
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "BOT:" in generated:
            resp = generated.split("BOT:")[-1].strip()
        else:
            resp = generated
        return resp

if __name__ == "__main__":
    bot = VillageBot()
    print("VillageConnect Chatbot (type 'exit' to quit, 'reset' to reset context)")
    while True:
        u = input("You: ").strip()
        if u.lower() in ("exit", "quit"):
            break
        if u.lower() == "reset":
            bot.reset()
            print("Context reset.")
            continue
        resp = bot.chat(u)
        print("Bot:", resp)
