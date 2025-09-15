import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

MODEL_NAME = os.environ.get("BASE_MODEL", "microsoft/DialoGPT-small")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./models/villageconnect-dialo")
DATA_PATH = os.environ.get("DATA_PATH", "../data/sample_conversations.jsonl")

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def preprocess(example):
        return {"text": example["text"]}
    dataset = dataset.map(preprocess)

    block_size = 512
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=block_size)
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        evaluation_strategy="no",
        warmup_steps=50,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete. Model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
