import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, # For 8-bit/4-bit quantization later
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset # For loading data
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training # For LoRA later
import os
import traceback

print("fine_tune.py: Top-level imports loaded.")

def load_model_for_finetuning(model_name_or_path="m-a-p/ChatMusician", use_quantization=False):
    """Loads the base model for fine-tuning."""
    print(f"load_model_for_finetuning: Loading base model {model_name_or_path} for fine-tuning...")

    # Quantization config (optional, for later)
    bnb_config = None
    if use_quantization:
        # Example for 4-bit, can be adjusted
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # or torch.float16
            bnb_4bit_use_double_quant=True,
        )
        print("load_model_for_finetuning: Using 4-bit quantization.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config, # Pass config if using quantization
            torch_dtype=torch.float16 if not use_quantization else None, # dtype handled by bnb_config
            device_map="auto", # Let transformers handle device placement
            trust_remote_code=True # If model requires it (ChatMusician does for tokenizer)
        )
        # No .eval() here, model is for training
        print(f"load_model_for_finetuning: Model loaded to device: {model.device}")

        # For some quantization setups, especially with LoRA:
        # if use_quantization:
        # model = prepare_model_for_kbit_training(model)
        
    except Exception as e:
        print(f"load_model_for_finetuning: ERROR loading model - {e}")
        traceback.print_exc()
        raise
    
    # Tokenizer (usually doesn't need quantization considerations for loading)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        # Set padding token if not present (important for training)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("load_model_for_finetuning: Tokenizer pad_token set to eos_token.")
        # For models like LLaMA, pad on the right
        tokenizer.padding_side = "right" 
        print("load_model_for_finetuning: Tokenizer loaded and configured.")
    except Exception as e:
        print(f"load_model_for_finetuning: ERROR loading tokenizer - {e}")
        traceback.print_exc()
        raise
        
    return model, tokenizer

def load_and_prepare_dataset(tokenizer, dataset_path: str, max_length=1024): # max_length can be tuned
    """Loads dataset from JSONL, tokenizes, and prepares for Causal LM training."""
    print(f"load_and_prepare_dataset: Loading dataset from {dataset_path}...")
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        print(f"load_and_prepare_dataset: Dataset loaded. Number of examples: {len(dataset)}")
    except Exception as e:
        print(f"load_and_prepare_dataset: ERROR loading dataset - {e}")
        traceback.print_exc()
        raise

    def tokenize_function(examples):
        # For Causal LM, we usually concatenate prompt and completion for the input,
        # and the labels are the same sequence, but we only calculate loss on the completion part.
        # The prompt structure is: "Human: {instruction}\n{abc_start}</s> Assistant: {abc_completion}</s>"
        # The 'prompt' field from JSONL already contains "Human: ... </s> Assistant: "
        # The 'completion' field from JSONL contains the rest of the ABC.
        # We need to combine them and add an EOS token at the very end of the completion.
        
        # texts = [p + c + tokenizer.eos_token for p, c in zip(examples["prompt"], examples["completion"])]
        
        # More robust approach for labels:
        # Tokenize prompts and completions separately to know where completion starts for labels.
        
        full_texts = []
        for i in range(len(examples["prompt"])):
            # The prompt field should already end with "Assistant: "
            # The completion field is the actual assistant's response.
            # We want the model to learn to predict the completion.
            # So, input_ids = tokenize(prompt + completion + eos)
            # labels = input_ids, but with prompt part masked out (-100)
            
            prompt_part = examples["prompt"][i]
            completion_part = examples["completion"][i]
            
            # This is the text the model sees as input and learns to predict from
            text_to_tokenize = prompt_part + completion_part + tokenizer.eos_token
            full_texts.append(text_to_tokenize)

        # Tokenize the full combined texts
        tokenized_inputs = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length", # or False, data_collator will handle
            # return_overflowing_tokens=True, # If you want to handle long sequences by splitting
            # return_length=True,
        )

        labels = tokenized_inputs["input_ids"].copy()

        # Mask out the prompt part from the labels
        # For each example, find where the completion part starts.
        # The prompt part in the tokenized sequence should not contribute to the loss.
        # The prompt (e.g., "Human: ... </s> Assistant: ") is what the model conditions on.
        # The completion is what it should learn to predict.
        
        # This is a common way: create labels that are a copy of input_ids.
        # Then, for the part of the input_ids that corresponds to the original prompt,
        # set the labels to -100 (which is ignored by the loss function).
        
        for i in range(len(examples["prompt"])):
            prompt_only_tokenized = tokenizer(examples["prompt"][i], add_special_tokens=False) # No EOS for just prompt
            prompt_length = len(prompt_only_tokenized["input_ids"])
            
            # Mask the prompt tokens in the labels
            for j in range(prompt_length):
                if j < max_length: # Ensure we don't go out of bounds if prompt is too long
                    labels[i][j] = -100
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("load_and_prepare_dataset: Tokenizing dataset...")
    processed_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print("load_and_prepare_dataset: Dataset tokenized and processed.")
    return processed_dataset

def main_fine_tune(
    model_name_or_path="m-a-p/ChatMusician",
    dataset_path="path/to/your/data_lead_dataset.jsonl", # This will come from Data Lead
    output_dir="./fine_tuned_continuoAI_v1",
    epochs=3, # Start with a small number for initial tests
    batch_size=1, # Adjust based on GPU VRAM. 1 is safest for large models.
    learning_rate=2e-5, # Common starting point for fine-tuning
    max_seq_length=1024, # Match with data processing
    use_quantization_for_training=False, # Set to True to test quantization
    # use_lora=False # Placeholder for LoRA
):
    print(f"main_fine_tune: Starting fine-tuning process...")
    print(f"main_fine_tune: Model: {model_name_or_path}, Dataset: {dataset_path}, Output: {output_dir}")

    model, tokenizer = load_model_for_finetuning(model_name_or_path, use_quantization=use_quantization_for_training)

    # --- LoRA/PEFT setup (placeholder, for later if needed) ---
    # if use_lora:
    #     print("main_fine_tune: Applying LoRA/PEFT modifications...")
    #     lora_config = LoraConfig(
    #         r=16, # Rank
    #         lora_alpha=32,
    #         target_modules=["q_proj", "v_proj"], # Example, find target modules for ChatMusician
    #         lora_dropout=0.05,
    #         bias="none",
    #         task_type="CAUSAL_LM"
    #     )
    #     model = get_peft_model(model, lora_config)
    #     model.print_trainable_parameters()
    # --- End LoRA/PEFT setup ---

    if not os.path.exists(dataset_path):
        print(f"main_fine_tune: ERROR - Dataset path not found: {dataset_path}")
        print("main_fine_tune: Please ensure the Data Lead has provided the dataset.")
        return

    train_dataset = load_and_prepare_dataset(tokenizer, dataset_path, max_length=max_seq_length)
    
    # Data collator for causal language modeling.
    # It handles padding and creating attention masks if padding was set to False during tokenization.
    # If padding="max_length" was used in tokenize_function, this mainly just converts lists to tensors.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # mlm=False for Causal LM

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Accumulate gradients if batch_size is small
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=10, # Log every 10 steps
        save_strategy="epoch", # Save a checkpoint at the end of each epoch
        # evaluation_strategy="epoch", # If you have an eval dataset
        # load_best_model_at_end=True, # If using evaluation
        fp16=True if not use_quantization_for_training and torch.cuda.is_available() else False, # Use mixed precision if not quantizing and on GPU
        # optim="paged_adamw_8bit" if use_quantization_for_training else "adamw_hf", # Paged optimizer for quantization
        report_to="tensorboard", # or "wandb", "none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, # If you create one
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("main_fine_tune: Starting training...")
    try:
        trainer.train()
        print("main_fine_tune: Training finished.")
        
        print(f"main_fine_tune: Saving model to {output_dir}...")
        trainer.save_model() # Saves the final model
        # tokenizer.save_pretrained(output_dir) # Save tokenizer if it was modified (usually not for just pad token)
        print("main_fine_tune: Model saved.")
    except Exception as e:
        print(f"main_fine_tune: ERROR during training - {e}")
        traceback.print_exc()

if __name__ == '__main__':
    print("fine_tune.py: Running main_fine_tune function for testing...")
    # YOU WILL NEED TO CREATE A DUMMY/SAMPLE JSONL FILE TO TEST THIS SCRIPT
    # For example, create a file `sample_data.jsonl` with 2-3 examples
    # in the format expected by load_and_prepare_dataset.
    # Example `sample_data.jsonl` line:
    # {"prompt": "Human: Continue this: X:1 K:C C D E</s> Assistant: ", "completion": "F G A | G2 z2"}

    # Create a dummy dataset file for initial testing if Data Lead hasn't provided one yet
    dummy_dataset_path = "dummy_music_data.jsonl"
    if not os.path.exists(dummy_dataset_path):
        print(f"Creating dummy dataset at {dummy_dataset_path} for testing fine_tune.py...")
        with open(dummy_dataset_path, "w") as f:
            f.write('{"prompt": "Human: Complete this tune: X:1 K:C C D E F | G A B c</s> Assistant: ", "completion": "d e f g | a b c2 |"}\n')
            f.write('{"prompt": "Human: Continue this melody: X:1 K:G G2 A2 | B2 c2</s> Assistant: ", "completion": "d2 e2 | f2 g2 | a4"}\n')
        print("Dummy dataset created.")

    # Before running, ensure you have GPU access or it will be extremely slow.
    # Adjust batch_size and epochs for quick testing.
    # Set use_quantization_for_training=True here if you want to test that path.
    main_fine_tune(
        dataset_path=dummy_dataset_path, 
        output_dir="./dummy_finetuned_model",
        epochs=1, 
        batch_size=1
    )
    print("fine_tune.py: End of main_fine_tune test run.")