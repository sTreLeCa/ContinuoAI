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

# (Continuing llm_module/fine_tune.py)

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

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if not use_quantization and bnb_config is None else None,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"FN: Model loaded. Device map: {model.hf_device_map}")
    except Exception as e:
        print(f"FN: ERROR loading model - {e}")
        traceback.print_exc()
        raise
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            print("FN: Tokenizer missing pad_token. Adding new pad_token from eos_token.")
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            model.resize_token_embeddings(len(tokenizer))
            print(f"FN: Resized model token embeddings to: {len(tokenizer)}")
        tokenizer.padding_side = "right"
        print("FN: Tokenizer loaded and configured.")
    except Exception as e:
        print(f"FN: ERROR loading tokenizer - {e}")
        traceback.print_exc()
        raise
            
    return model, tokenizer

# (Continuing llm_module/fine_tune.py)

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

    print("FN: Tokenizing dataset...")
    processed_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset"
    )
    print("FN: Dataset tokenized.")
    return processed_dataset
    
    # (Continuing llm_module/fine_tune.py)

def main_fine_tune(
    model_name_or_path="m-a-p/ChatMusician",
    dataset_path="dummy_music_data.jsonl", # Default to dummy for easy testing
    output_dir="./fine_tuned_model_output", # Generic output dir
    epochs=1, # Keep low for initial/dummy tests
    batch_size=1, 
    gradient_accumulation_steps=4,
    learning_rate=2e-4, # Slightly higher for LoRA often works
    max_seq_length=512, # Shorter for dummy tests to speed up tokenization
    use_quantization_for_training=True,
    use_lora=True
):
    print(f"FN: Starting fine-tuning process...")
    print(f"FN: Config - Model={model_name_or_path}, Dataset={dataset_path}, Output={output_dir}")
    print(f"FN: Config - Epochs={epochs}, BS={batch_size}, GradAccum={gradient_accumulation_steps}, LR={learning_rate}")
    print(f"FN: Config - MaxSeqLen={max_seq_length}, Quantize={use_quantization_for_training}, LoRA={use_lora}")

    model, tokenizer = load_model_for_finetuning(model_name_or_path, use_quantization=use_quantization_for_training)

    if use_quantization_for_training: # Should be True if bnb_config was used
        print("FN: Preparing k-bit model for training (gradient checkpointing, etc.)...")
        model = prepare_model_for_kbit_training(model) # Important for QLoRA

    if use_lora:
        print("FN: Applying LoRA/PEFT modifications...")
        # For LLaMA-like models, common targets are q_proj, v_proj.
        # Can expand to k_proj, o_proj, gate_proj, up_proj, down_proj for potentially better results.
        # Start simple and verify it trains.
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"], # ADJUST THESE based on ChatMusician's actual layer names if needed
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        print("FN: LoRA model created.")
        model.print_trainable_parameters()
    
    if not os.path.exists(dataset_path):
        print(f"FN: ERROR - Dataset path not found: {dataset_path}")
        return

    train_dataset = load_and_prepare_dataset(tokenizer, dataset_path, max_length=max_seq_length)

    # (Continuing main_fine_tune function in llm_module/fine_tune.py)

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
        fp16=False if use_quantization_for_training or not torch.cuda.is_available() or torch.cuda.is_bf16_supported() else True, # fp16 if not quantizing AND no bfloat16
        bf16=True if not use_quantization_for_training and torch.cuda.is_bf16_supported() else False, # bf16 if not quantizing AND supported
        gradient_checkpointing=True, # Often beneficial, especially with PEFT
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
        # If using LoRA, this saves the adapter. The base model isn't resaved here.
        # Tokenizer is usually saved if it was part of the PEFT model or if you want to ensure consistency.
        # tokenizer.save_pretrained(final_save_path) 
        print(f"FN: Model/adapters saved to {final_save_path}.")
        
    except Exception as e:
        print(f"FN: ERROR during training or saving - {e}")
        traceback.print_exc()

        # (Continuing llm_module/fine_tune.py)

if __name__ == '__main__':
    print("\n" + "="*30)
    print("fine_tune.py: Running main_fine_tune function for initial testing...")
    print("="*30 + "\n")
    
    dummy_dataset_path = "dummy_music_data.jsonl"
    if not os.path.exists(dummy_dataset_path):
        print(f"Creating dummy dataset at {dummy_dataset_path} for testing fine_tune.py...")
        with open(dummy_dataset_path, "w", encoding="utf-8") as f:
            f.write('{"prompt": "Human: Complete this tune: X:1 K:C C D E F | G A B c</s> Assistant: ", "completion": "d e f g | a b c2 |"}\n')
            f.write('{"prompt": "Human: Continue this melody: X:1 K:G G2 A2 | B2 c2</s> Assistant: ", "completion": "d2 e2 | f2 g2 | a4"}\n')
            f.write('{"prompt": "Human: Finish this phrase: M:C L:1/8 K:Am A B c d | e f g a</s> Assistant: ", "completion": "b c\' d\' e\' | f\'2 g\'2 a\'2 |"}\n')
        print("Dummy dataset created.")

    # Test with quantization and LoRA enabled
    main_fine_tune(
        model_name_or_path="m-a-p/ChatMusician",
        dataset_path=dummy_dataset_path, 
        output_dir="./dummy_qlora_finetuned_output", # New output dir for this test
        epochs=1, 
        batch_size=1, 
        gradient_accumulation_steps=1, 
        max_seq_length=512, 
        use_quantization_for_training=True,
        use_lora=True 
    )
    print("\n" + "="*30)
    print("fine_tune.py: End of main_fine_tune test run.")
    print("="*30 + "\n")