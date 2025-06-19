# llm_module/inference_engine.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig # For optional quantization of base model during inference
)
from peft import PeftModel # For loading LoRA adapters
import traceback
import os # To check if adapter path exists

print("inference_engine.py: Top-level imports loaded.")

def load_model_for_inference(
    base_model_name="m-a-p/ChatMusician",
    adapter_path=None,
    use_quantization_for_base_model_inference=False # Default to False for higher quality base inference
                                                    # Set to True if VRAM is extremely tight even for inference,
                                                    # or if adapters were trained on a quantized base
                                                    # and you want max consistency.
):
    """Loads the base model and optionally applies PEFT LoRA adapters for inference."""
    print(f"INFER: Loading model. Base: {base_model_name}, Adapters: {adapter_path}")
    print(f"INFER: Quantize base model for inference: {use_quantization_for_base_model_inference}")

    bnb_config_inference = None
    if use_quantization_for_base_model_inference:
        print("INFER: Configuring 4-bit quantization for base model inference...")
        bnb_config_inference = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    print(f"INFER: Attempting to load base model '{base_model_name}'...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config_inference,
            torch_dtype=torch.float16 if not use_quantization_for_base_model_inference and bnb_config_inference is None else None,
            device_map="auto",
            trust_remote_code=True,
            # Do NOT call .eval() yet if we are applying adapters, PeftModel will handle it.
        )
        print(f"INFER: Base model '{base_model_name}' loaded. Device map: {base_model.hf_device_map if hasattr(base_model, 'hf_device_map') else 'N/A'}")
    except Exception as e:
        print(f"INFER: ERROR loading base model - {e}")
        traceback.print_exc()
        raise

    model_to_use = base_model
    if adapter_path and os.path.exists(adapter_path): # Check if path exists
        print(f"INFER: Attempting to load LoRA adapters from {adapter_path}...")
        try:
            # Ensure the base model is not in .eval() mode if it was set before PeftModel loads
            if hasattr(base_model, 'training'): # Check if it's a PreTrainedModel instance
                 base_model.train(False) # Set to eval mode temporarily if it's not done by PeftModel
                                         # Actually, PeftModel expects the base model NOT to be in .eval() during loading.
                                         # So, we ensure it's not in eval() mode before PeftModel.from_pretrained
            
            # If the base model was quantized, PEFT needs to be aware.
            # For QLoRA, the base model IS quantized.
            model_to_use = PeftModel.from_pretrained(base_model, adapter_path)
            print("INFER: LoRA adapters loaded successfully onto base model.")
            
            # Optional: Merge adapters for faster inference if VRAM allows.
            # This creates a new model with merged weights and unloads PEFT wrappers.
            # After merging, the model behaves like a standard Hugging Face model.
            # Consider making this a configurable option.
            # print("INFER: Merging LoRA adapters into the base model for optimized inference...")
            # model_to_use = model_to_use.merge_and_unload()
            # print("INFER: Adapters merged and PEFT model unloaded.")

        except Exception as e:
            print(f"INFER: ERROR loading LoRA adapters from {adapter_path}: {e}")
            print("INFER: Will proceed with the base model only.")
            traceback.print_exc()
            # model_to_use remains base_model
    elif adapter_path: # Path provided but does not exist
        print(f"INFER: WARNING - Adapter path '{adapter_path}' provided but not found. Using base model.")
    else:
        print("INFER: No adapter path provided, using base model.")
    
    model_to_use.eval() # Set final model to evaluation mode
    print(f"INFER: Final model is ready for inference on device: {model_to_use.device if hasattr(model_to_use, 'device') else 'N/A'}.")

    print(f"INFER: Attempting to load tokenizer for {base_model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            # Match the setup in fine_tune.py if pad_token was added there
            # If tokenizer was saved with fine-tuned model, it would have the pad token.
            # For now, assuming we always use base tokenizer and fix pad_token if needed.
            print("INFER: Tokenizer missing pad_token. Setting pad_token = eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 
        print("INFER: Tokenizer loaded and configured.")
    except Exception as e:
        print(f"INFER: ERROR loading tokenizer - {e}")
        traceback.print_exc()
        raise
            
    return model_to_use, tokenizer

# --- Global Model and Tokenizer ---
MODEL = None
TOKENIZER = None
MODEL_LOAD_ERROR = None

# --- !! CONFIGURATION FOR INFERENCE !! ---
# Set to None to use the base m-a-p/ChatMusician model
# Set to the path of your fine-tuned LoRA adapter checkpoint directory to use the fine-tuned model
# Example: ADAPTER_CHECKPOINT_PATH = "./dummy_qlora_finetuned_output/final_checkpoint/"
# ADAPTER_CHECKPOINT_PATH = "./continuoAI_finetuned_thesession_v1_sample/final_checkpoint/" 
ADAPTER_CHECKPOINT_PATH = None # << UNCOMMENT THIS LINE TO TEST WITH BASE MODEL

# If your LoRA adapters were trained on a quantized base model (QLoRA),
# it's generally best to also load the base model quantized for inference with those adapters.
USE_QUANTIZATION_FOR_BASE_INFERENCE = True if ADAPTER_CHECKPOINT_PATH else False
# (If ADAPTER_CHECKPOINT_PATH is None, quantization for base is less critical unless VRAM is an issue)

print("inference_engine.py: Attempting to load global MODEL and TOKENIZER (with potential adapters)...")
try:
    MODEL, TOKENIZER = load_model_for_inference(
        adapter_path=ADAPTER_CHECKPOINT_PATH,
        use_quantization_for_base_model_inference=USE_QUANTIZATION_FOR_BASE_INFERENCE
    )
    print("inference_engine.py: Global MODEL and TOKENIZER loaded successfully.")
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    print(f"inference_engine.py: CRITICAL ERROR during global model load: {MODEL_LOAD_ERROR}")

def generate_music_continuation(prompt_abc: str,
                                max_new_tokens=256,
                                temperature=0.7,
                                top_k=40,
                                top_p=0.9,
                                repetition_penalty=1.1):
    """
    Generates a music continuation from an ABC prompt string using the globally loaded model.
    """
    if MODEL is None or TOKENIZER is None:
        error_message = f"generate_music_continuation: Model or Tokenizer not loaded. Load error: {MODEL_LOAD_ERROR}"
        print(error_message)
        return error_message

    print(f"generate_music_continuation: Generating continuation for prompt (first 80 chars): '{prompt_abc[:80]}...'")

    try:
        pad_token_id = TOKENIZER.pad_token_id if TOKENIZER.pad_token_id is not None else TOKENIZER.eos_token_id

        generation_config = GenerationConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_beams=1,
            repetition_penalty=repetition_penalty,
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=pad_token_id,
            min_new_tokens=20,
            max_new_tokens=max_new_tokens,
        )

        inputs = TOKENIZER(prompt_abc, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(MODEL.device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(MODEL.device)

        print(f"generate_music_continuation: Input ID shape: {input_ids.shape}, Device: {input_ids.device}")
        
        with torch.no_grad():
            print("generate_music_continuation: Calling model.generate()...")
            response_ids = MODEL.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
            print("generate_music_continuation: model.generate() finished.")

        generated_tokens_only = response_ids[0][input_ids.shape[1]:]
        response_text = TOKENIZER.decode(generated_tokens_only, skip_special_tokens=True)
        
        print("generate_music_continuation: Generation complete. Decoded response.")
        return response_text

    except Exception as e:
        error_message = f"generate_music_continuation: ERROR during generation: {e}"
        print(error_message)
        traceback.print_exc()
        return error_message

# This block will only execute when the script is run directly
if __name__ == "__main__":
    print("\n" + "="*30)
    print("Running inference_engine.py as main script for testing...")
    print(f"Testing with adapters: {ADAPTER_CHECKPOINT_PATH}")
    print("="*30 + "\n")

    if MODEL and TOKENIZER:
        # --- Test Case 1: Chord Progression (Less relevant for evaluating completion-focused adapters) ---
        # instruction1 = """Develop a musical piece using the given chord progression.
        # 'Dm', 'C', 'Dm', 'Dm', 'C', 'Dm', 'C', 'Dm'"""
        # prompt1 = f"Human: {instruction1}</s> Assistant: "
        # print(f"\n--- Test 1: Chord Progression ---")
        # print(f"Prompting with:\n{prompt1}")
        # continuation1 = generate_music_continuation(prompt1, max_new_tokens=350)
        # print("\nGenerated Music (Chord Progression):")
        # print(continuation1)
        # print("---------------------------------")

        # --- Test Case 2: ABC Continuation (Focus on this for your fine-tuned model) ---
        # Use a prompt similar to what was in your fine-tuning data
        abc_start_snippet_for_test = """X:1
T:Test Completion Start
M:4/4
L:1/8
K:Gmaj
G A B c | d2 e2""" # Shorter prompt to see what it does
        
        instruction2 = f"Here is the beginning of a musical piece. Please continue it in a coherent style:\n{abc_start_snippet_for_test}"
        # If your fine-tuning prompts were simpler, e.g., just "Human: [ABC_START]</s> Assistant: ", adjust here.
        # Let's assume the fine-tuning prompt included the instruction text.
        prompt2 = f"Human: {instruction2}</s> Assistant: "

        print(f"\n--- Test 2: ABC Continuation (with fine-tuned adapters if path set) ---")
        print(f"Prompting with:\n{prompt2}")
        continuation2 = generate_music_continuation(prompt2, max_new_tokens=128) # Generate a shorter continuation for quick test
        print("\nGenerated Music (ABC Continuation):")
        print(continuation2)
        print("--------------------------------")
        
        # --- Test Case 3: Another ABC Continuation example ---
        abc_start_snippet_for_test_2 = """X:1
T:Another Test
M:6/8
L:1/8
K:Dmaj
A | dcd FGA | Bcd"""
        instruction3 = f"Complete the following musical phrase:\n{abc_start_snippet_for_test_2}"
        prompt3 = f"Human: {instruction3}</s> Assistant: "
        print(f"\n--- Test 3: ABC Continuation 2 ---")
        print(f"Prompting with:\n{prompt3}")
        continuation3 = generate_music_continuation(prompt3, max_new_tokens=128)
        print("\nGenerated Music (ABC Continuation 2):")
        print(continuation3)
        print("--------------------------------")


    elif MODEL_LOAD_ERROR:
        print(f"Cannot run test: Model or Tokenizer failed to load. Error: {MODEL_LOAD_ERROR}")
    else:
        print("Cannot run test: Model or Tokenizer not initialized for an unknown reason.")

    print("\n" + "="*30)
    print("inference_engine.py: End of main script execution.")
    print("="*30 + "\n")