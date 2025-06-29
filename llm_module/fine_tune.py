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
    """
    Loads the base model for fine-tuning, with optional 4-bit quantization.
    Also loads the tokenizer and configures it for training (pad token).
    """
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
        print(f"FN: Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
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
    """
    Loads dataset from a JSONL file and tokenizes it for Causal LM fine-tuning.
    It masks the prompt part of the labels so loss is only calculated on the completion.
    """
    print(f"FN: Loading dataset from {dataset_path}...")
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        print(f"FN: Dataset loaded. Examples: {len(dataset)}")
    except Exception as e:
        print(f"FN: ERROR loading dataset - {e}")
        traceback.print_exc()
        raise

    def tokenize_function(examples):
        # Concatenate prompt and completion for the input sequence
        full_texts = [p + c + tokenizer.eos_token for p, c in zip(examples["prompt"], examples["completion"])]
        
        # Tokenize the full combined texts
        tokenized_full = tokenizer(
            full_texts,
            max_length=max_length,
            truncation=True,
            padding=False, # DataCollator will handle padding
            add_special_tokens=False
        )
        
        # Create labels that are a copy of input_ids
        tokenized_full["labels"] = tokenized_full["input_ids"].copy()

        # Mask out the prompt part from the labels
        prompt_lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in examples["prompt"]]
        for i, prompt_len in enumerate(prompt_lengths):
            for j in range(prompt_len):
                if j < max_length:
                    tokenized_full["labels"][i][j] = -100 # -100 is the ignore index for loss calculation
        
        return tokenized_full

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
    dataset_path="dummy_music_data.jsonl", # Default to dummy for safe, quick testing
    output_dir="./fine_tuned_model_output", # Generic default output dir
    epochs=1, # Default to 1 for quick test
    batch_size=1, 
    gradient_accumulation_steps=4,
    learning_rate=2e-4, # Common starting point for LoRA
    max_seq_length=512, # Shorter default for quick test
    use_quantization_for_training=True,
    use_lora=True
):
    """
    Main function to run the QLoRA fine-tuning process.
    """
    print(f"FN: Starting fine-tuning process...")
    print(f"FN: Config - Model={model_name_or_path}, Dataset={dataset_path}, Output={output_dir}")
    print(f"FN: Config - Epochs={epochs}, BS={batch_size}, GradAccum={gradient_accumulation_steps}, LR={learning_rate}")
    print(f"FN: Config - MaxSeqLen={max_seq_length}, Quantize={use_quantization_for_training}, LoRA={use_lora}")

    model, tokenizer = load_model_for_finetuning(model_name_or_path, use_quantization=use_quantization_for_training)

    if use_quantization_for_training:
        print("FN: Preparing k-bit model for training...")
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        print("FN: Applying LoRA/PEFT modifications...")
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"], # Common LLaMA targets, verify for ChatMusician if needed
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        print("FN: LoRA model created.")
        model.print_trainable_parameters()
    
    if not os.path.exists(dataset_path):
        print(f"FN: ERROR - Dataset path not found: {dataset_path}")
        print("FN: Please ensure the dataset file exists.")
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
    # This block allows you to run this script directly for testing or experiments.
    # To switch between tests, you can comment/uncomment the respective main_fine_tune() calls.

    # --- Option 1: Quick Structural Test with Dummy Data ---
    # This is the safe default for team members to verify the script runs.
    
    # print("\n" + "="*30)
    # print("fine_tune.py: Running main_fine_tune function for DUMMY DATA TEST...")
    # print("="*30 + "\n")
    
    # dummy_dataset_path = "dummy_music_data.jsonl"
    # if not os.path.exists(dummy_dataset_path):
    #     print(f"Creating dummy dataset at {dummy_dataset_path} for testing fine_tune.py...")
    #     with open(dummy_dataset_path, "w", encoding="utf-8") as f:
    #         f.write('{"prompt": "Human: Complete this tune: X:1 K:C C D E F | G A B c</s> Assistant: ", "completion": "d e f g | a b c2 |"}\n')
    #         f.write('{"prompt": "Human: Continue this melody: X:1 K:G G2 A2 | B2 c2</s> Assistant: ", "completion": "d2 e2 | f2 g2 | a4"}\n')
    #     print("Dummy dataset created.")
    
    # main_fine_tune(
    #     dataset_path=dummy_dataset_path, 
    #     output_dir="./dummy_qlora_finetuned_output",
    #     # Uses safe defaults from the function signature: epochs=1, max_seq_length=512, etc.
    # )

    # --- Option 2: Run Fine-Tuning with Real Sample Data ---
    # This is the experiment you are currently running.
    # To run this, uncomment this block and comment out the "Dummy Data Test" block above.
    
    print("\n" + "="*30)
    print("fine_tune.py: Running main_fine_tune function with REAL SAMPLE DATA...")
    print("="*30 + "\n")
    
    main_fine_tune(
        dataset_path="finetuning_dataset_v1_sample.jsonl",
        output_dir="./continuoAI_finetuned_thesession_v1_sample",
        epochs=3, 
        max_seq_length=1024, # Using a longer sequence length for more context
        learning_rate=2e-4,
        gradient_accumulation_steps=4,
        batch_size=1
    )

    print("\n" + "="*30)
    print("fine_tune.py: End of main test run.")
    print("="*30 + "\n")