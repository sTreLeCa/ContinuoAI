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

def load_and_prepare_model_and_tokenizer(model_name_or_path="m-a-p/ChatMusician", use_quantization=True):
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
    
    # Load the tokenizer first
    print(f"FN: Attempting to load tokenizer for {model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        print(f"FN: Tokenizer loaded. Initial vocab size: {len(tokenizer)}")
        
        # Set pad token properly
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print("FN: Set pad_token to eos_token")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print("FN: Added new pad_token")
        
        tokenizer.padding_side = "right"
        
    except Exception as e:
        print(f"FN: ERROR loading tokenizer - {e}")
        traceback.print_exc()
        raise

    print(f"FN: Attempting to load model {model_name_or_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        print(f"FN: Model loaded. Initial model vocab size: {model.config.vocab_size}")
    except Exception as e:
        print(f"FN: ERROR loading model - {e}")
        traceback.print_exc()
        raise

    # Handle vocab size mismatch
    current_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    
    if tokenizer_vocab_size != current_vocab_size:
        print(f"FN: Vocab size mismatch! Tokenizer: {tokenizer_vocab_size}, Model: {current_vocab_size}")
        print("FN: Resizing model token embeddings...")
        model.resize_token_embeddings(tokenizer_vocab_size)
        model.config.vocab_size = tokenizer_vocab_size
        print(f"FN: Resized model token embeddings to: {tokenizer_vocab_size}")
    else:
        print("FN: Tokenizer and model vocab sizes match.")

    if use_quantization:
        print("FN: Preparing k-bit model for training...")
        model = prepare_model_for_kbit_training(model)

    # Ensure critical parameters are trainable
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in ['embed_tokens', 'lm_head']):
            param.requires_grad = True

    return model, tokenizer

def load_and_prepare_dataset(tokenizer, dataset_path: str, max_length=1024):
    print(f"FN: Loading dataset from {dataset_path}...")
    
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(10))
        print(f"FN: Dataset loaded and shuffled. Examples: {len(dataset)}")
    except Exception as e:
        print(f"FN: ERROR loading dataset - {e}")
        raise

    def tokenize_function(examples):
        # Combine prompts and completions
        full_texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # Ensure strings are properly formatted
            prompt_str = str(prompt) if prompt is not None else ""
            completion_str = str(completion) if completion is not None else ""
            full_text = prompt_str + completion_str + tokenizer.eos_token
            full_texts.append(full_text)
        
        # Tokenize with proper settings
        tokenized = tokenizer(
            full_texts,
            max_length=max_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
            return_tensors=None  # Return lists, not tensors
        )
        
        # Create labels (copy of input_ids)
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        
        # Mask prompt tokens in labels (set to -100 so they're ignored in loss)
        for i, (prompt, completion) in enumerate(zip(examples["prompt"], examples["completion"])):
            prompt_str = str(prompt) if prompt is not None else ""
            
            # Get prompt length
            prompt_tokens = tokenizer(
                prompt_str,
                add_special_tokens=False,
                return_tensors=None
            )["input_ids"]
            prompt_len = len(prompt_tokens)
            
            # Mask prompt tokens in labels
            for j in range(min(prompt_len, len(tokenized["labels"][i]))):
                tokenized["labels"][i][j] = -100
        
        # Validate token IDs are within vocab range
        vocab_size = len(tokenizer)
        for i, input_ids in enumerate(tokenized["input_ids"]):
            # Check for invalid token IDs
            valid_ids = [id for id in input_ids if 0 <= id < vocab_size]
            if len(valid_ids) != len(input_ids):
                print(f"FN: WARNING - Found invalid token IDs in example {i}, filtering...")
                tokenized["input_ids"][i] = valid_ids
                tokenized["labels"][i] = tokenized["labels"][i][:len(valid_ids)]
        
        return tokenized

    print("FN: Tokenizing dataset...")
    try:
        processed_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Running tokenizer on dataset",
            num_proc=1  # Use single process to avoid issues
        )
        print("FN: Dataset tokenized and processed.")
        return processed_dataset
    except Exception as e:
        print(f"FN: ERROR during dataset tokenization - {e}")
        traceback.print_exc()
        raise

def main_fine_tune(
    model_name_or_path="m-a-p/ChatMusician",
    dataset_path="finetuning_dataset_large_v1.jsonl",
    output_dir="./continuoAI_finetuned_large_v1",
    epochs=3,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_seq_length=1024,
    use_quantization_for_training=True,
    use_lora=True
):
    print(f"FN: Starting fine-tuning process...")
    print(f"FN: Model: {model_name_or_path}")
    print(f"FN: Dataset: {dataset_path}")
    print(f"FN: Output: {output_dir}")
    print(f"FN: Epochs: {epochs}, Batch size: {batch_size}")
    print(f"FN: Learning rate: {learning_rate}")
    print(f"FN: Max sequence length: {max_seq_length}")
    print(f"FN: Quantization: {use_quantization_for_training}")
    print(f"FN: LoRA: {use_lora}")

    # Set environment variables for better CUDA debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    try:
        # Load model and tokenizer
        model, tokenizer = load_and_prepare_model_and_tokenizer(
            model_name_or_path, 
            use_quantization=use_quantization_for_training
        )

        if use_lora:
            print("FN: Applying LoRA/PEFT modifications...")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model = get_peft_model(model, lora_config)
            print("FN: LoRA model created.")
            model.print_trainable_parameters()
        
        # Check dataset exists
        if not os.path.exists(dataset_path):
            print(f"FN: ERROR - Dataset path not found: {dataset_path}")
            return

        # Load and prepare dataset
        train_dataset = load_and_prepare_dataset(tokenizer, dataset_path, max_length=max_seq_length)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False,
            pad_to_multiple_of=8  # Helps with efficiency
        )

        # Training arguments with safer settings
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_strategy="steps",
            logging_steps=25,
            save_strategy="epoch",
            save_steps=500,
            evaluation_strategy="no",
            fp16=False,  # Disable FP16 to avoid precision issues
            bf16=False,  # Disable BF16 as well for stability
            gradient_checkpointing=True,
            optim="paged_adamw_8bit" if use_quantization_for_training else "adamw_torch",
            report_to=None,  # Disable reporting to avoid issues
            dataloader_drop_last=True,  # Drop incomplete batches
            dataloader_num_workers=0,  # Use main process only
            remove_unused_columns=False,
            max_grad_norm=1.0,  # Gradient clipping
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        print("FN: Starting training (trainer.train())...")
        trainer.train()
        print("FN: Training finished successfully.")
        
        # Save final model
        final_save_path = os.path.join(output_dir, "final_checkpoint")
        print(f"FN: Saving final model/adapters to {final_save_path}...")
        trainer.save_model(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        print(f"FN: Model/adapters saved to {final_save_path}.")
        
    except Exception as e:
        print(f"FN: ERROR during training or saving - {e}")
        traceback.print_exc()
        raise

if __name__ == '__main__':
    print("\n" + "="*30)
    print("fine_tune.py: Running main fine-tuning script...")
    print("="*30 + "\n")
    
    try:
        main_fine_tune()
    except Exception as e:
        print(f"FN: FATAL ERROR - {e}")
        traceback.print_exc()
    
    print("\n" + "="*30)
    print("fine_tune.py: End of run.")
    print("="*30 + "\n")