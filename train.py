import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import yaml
from datasets import load_dataset
import wandb
from peft import LoraConfig, get_peft_model

def load_config():
    with open("config/model_config.yaml", "r") as f:
        return yaml.safe_load(f)

def prepare_medical_prompt(question):
    """Prepare medical question with appropriate context and constraints."""
    return f"""Question: {question}
    
    Please provide a helpful response that:
    - Explains general information about the condition
    - Suggests lifestyle changes or home remedies if applicable
    - Recommends when to seek professional medical help
    
    Response:"""

def prepare_dataset(config):
    """Load and prepare the medical QA dataset."""
    # Move tokenizer creation to beginning and return it
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("json", 
                         data_files={
                             "train": config["dataset"]["train_path"],
                             "validation": config["dataset"]["eval_path"]
                         },
                         field="data")
    
    def tokenize_function(examples):
        prompts = [prepare_medical_prompt(q) for q in examples["question"]]
        responses = examples["answer"]
        
        # Combine prompts and responses
        combined_texts = [p + r for p, r in zip(prompts, responses)]
        
        return tokenizer(
            combined_texts,
            padding="max_length",
            truncation=True,
            max_length=config["model"]["max_length"]
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset, tokenizer

def main():
    # Load configuration
    config = load_config()
    
    # Initialize wandb for experiment tracking
    wandb.init(project="medical-slm", config=config)
    device_map = "mps" 
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.float16,
        device_map=device_map,
        
    )
    
    # Configure LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Prepare dataset - Modified to receive both returns
    dataset, tokenizer = prepare_dataset(config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./medical_slm_output",
        per_device_train_batch_size=config["training"]["batch_size"],
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=config["training"]["num_epochs"],
        warmup_steps=config["training"]["warmup_steps"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        weight_decay=config["training"]["weight_decay"],
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        report_to="wandb"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model("./medical_slm_final")
    wandb.finish()

if __name__ == "__main__":
    main() 