# llm_module/fine_tune.py

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import os
import traceback

print("fine_tune.py: Top-level imports loaded.")

def load_model_for_finetuning(model_name_or_path="m-a-p/ChatMusician", use_quantization=True):
    print(f"FN: Loading base model {model_name_or_path} for fine-tuning (Quantization: {use_quantization})...")

    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("FN: 4-bit quantization configured.")
    else:
        print("FN: Quantization not enabled for model loading.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if not use_quantization and bnb_config is None else None,
            device_map="auto",
            trust_remote_code=True # Necessary for some models/tokenizers
        )
        # No .eval() here as model is for training
        print(f"FN: Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A (not a HF model or specific device)'}")
    except Exception as e:
        print(f"FN: ERROR loading model - {e}")
        traceback.print_exc()
        raise
    
    print(f"FN: Attempting to load tokenizer for {model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            print("FN: Tokenizer missing pad_token. Adding new pad_token from eos_token.")
            # Add padding token and resize model embeddings
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            model.resize_token_embeddings(len(tokenizer)) 
            print(f"FN: Resized model token embeddings to: {len(tokenizer)}")
        
        # For LLaMA-based models, padding on the right is standard for training.
        tokenizer.padding_side = "right" 
        print("FN: Tokenizer loaded and configured (pad_token set, padding_side='right').")
    except Exception as e:
        print(f"FN: ERROR loading tokenizer - {e}")
        traceback.print_exc()
        raise
            
    return model, tokenizer

def load_and_prepare_dataset(tokenizer, dataset_path: str, max_length=1024):
    print(f"FN: Loading dataset from {dataset_path}...")
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        print(f"FN: Dataset loaded. Examples: {len(dataset)}")
    except Exception as e:
        print(f"FN: ERROR loading dataset - {e}")
        traceback.print_exc()
        raise

    def tokenize_function(examples):
        processed_examples = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(examples["prompt"])):
            prompt_text = examples["prompt"][i]
            completion_text = examples["completion"][i]
            full_text = prompt_text + completion_text + tokenizer.eos_token
            
            tokenized_full = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding=False, 
                add_special_tokens=False 
            )
            labels = tokenized_full["input_ids"].copy()
            tokenized_prompt_only = tokenizer(prompt_text, add_special_tokens=False)
            prompt_token_length = len(tokenized_prompt_only["input_ids"])

            for j in range(prompt_token_length):
                if j < len(labels): 
                    labels[j] = -100 
            
            processed_examples["input_ids"].append(tokenized_full["input_ids"])
            processed_examples["attention_mask"].append(tokenized_full["attention_mask"])
            processed_examples["labels"].append(labels)
        return processed_examples

    print("FN: Tokenizing dataset (this might take a while for large datasets)...")
    processed_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset"
    )
    print("FN: Dataset tokenized and processed.")
    return processed_dataset

def main_fine_tune(
    model_name_or_path="m-a-p/ChatMusician",
    dataset_path="finetuning_dataset_v1_sample.jsonl", # Default to the real sample data
    output_dir="./continuoAI_finetuned_thesession_v1_sample", # Descriptive output dir
    epochs=3, 
    batch_size=1, 
    gradient_accumulation_steps=4,
    learning_rate=2e-4, 
    max_seq_length=1024, 
    use_quantization_for_training=True,
    use_lora=True
):
    print(f"FN: Starting fine-tuning process...")
    print(f"FN: Config - Model={model_name_or_path}, Dataset={dataset_path}, Output={output_dir}")
    print(f"FN: Config - Epochs={epochs}, BS={batch_size}, GradAccum={gradient_accumulation_steps}, LR={learning_rate}")
    print(f"FN: Config - MaxSeqLen={max_seq_length}, Quantize={use_quantization_for_training}, LoRA={use_lora}")

    model, tokenizer = load_model_for_finetuning(model_name_or_path, use_quantization=use_quantization_for_training)

    if use_quantization_for_training:
        print("FN: Preparing k-bit model for training (enables gradient checkpointing, etc.)...")
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        print("FN: Applying LoRA/PEFT modifications...")
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"], # Start simple, verify these exist in ChatMusician
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        print("FN: LoRA model created.")
        model.print_trainable_parameters()
    
    if not os.path.exists(dataset_path):
        print(f"FN: ERROR - Dataset path not found: {dataset_path}")
        print("FN: Please ensure the dataset file exists (e.g., finetuning_dataset_v1_sample.jsonl).")
        return

    train_dataset = load_and_prepare_dataset(tokenizer, dataset_path, max_length=max_seq_length)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("FN: Configuring TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        fp16=False if use_quantization_for_training or not torch.cuda.is_available() or torch.cuda.is_bf16_supported() else True,
        bf16=True if not use_quantization_for_training and torch.cuda.is_bf16_supported() else False,
        gradient_checkpointing=True, 
        optim="paged_adamw_8bit" if use_quantization_for_training and use_lora else "adamw_torch",
        report_to="tensorboard",
    )

    print("FN: Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("FN: Starting training (trainer.train())...")
    try:
        trainer.train()
        print("FN: Training finished successfully.")
        
        final_save_path = os.path.join(output_dir, "final_checkpoint")
        print(f"FN: Saving final model/adapters to {final_save_path}...")
        trainer.save_model(final_save_path) 
        print(f"FN: Model/adapters saved to {final_save_path}.")
        
    except Exception as e:
        print(f"FN: ERROR during training or saving - {e}")
        traceback.print_exc()

if __name__ == '__main__':
    print("\n" + "="*30)
    print("fine_tune.py: Running main_fine_tune function...")
    print("="*30 + "\n")
    
    # This will now use the real sample dataset generated by preprocess_data.py
    main_fine_tune() # Uses default parameters defined in the function signature

    print("\n" + "="*30)
    print("fine_tune.py: End of main_fine_tune run.")
    print("="*30 + "\n")