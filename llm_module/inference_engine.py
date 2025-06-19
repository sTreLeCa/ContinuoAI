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
    use_quantization_for_base_model_inference=False
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
            device_map="auto", # Ensures model is placed on GPU if possible
            trust_remote_code=True,
            # Do NOT call .eval() yet if we are applying adapters, PeftModel will handle it.
        )
        print(f"INFER: Base model '{base_model_name}' loaded. Device map: {base_model.hf_device_map if hasattr(base_model, 'hf_device_map') else 'N/A'}")
    except Exception as e:
        print(f"INFER: ERROR loading base model - {e}")
        traceback.print_exc()
        raise

    model_to_use = base_model
    if adapter_path and os.path.isdir(adapter_path): # Check if adapter_path is a valid directory
        print(f"INFER: Attempting to load LoRA adapters from {adapter_path}...")
        try:
            # When loading a PEFT model, the base model should ideally not be in .eval() mode yet.
            # PeftModel.from_pretrained will handle setting the correct mode.
            # If base_model was quantized, PEFT handles it.
            model_to_use = PeftModel.from_pretrained(base_model, adapter_path)
            print("INFER: LoRA adapters loaded successfully onto base model.")
            
            # Optional: Merge adapters for potentially faster inference.
            # This replaces the LoRA layers with their merged equivalents.
            # print("INFER: Merging LoRA adapters into the base model for optimized inference...")
            # model_to_use = model_to_use.merge_and_unload()
            # print("INFER: Adapters merged and PEFT model unloaded.")

        except Exception as e:
            print(f"INFER: ERROR loading LoRA adapters from {adapter_path}: {e}")
            print("INFER: Will proceed with the base model only.")
            traceback.print_exc()
            # model_to_use remains base_model
    elif adapter_path: # Path provided but does not exist or is not a directory
        print(f"INFER: WARNING - Adapter path '{adapter_path}' provided but not found or not a directory. Using base model.")
    else:
        print("INFER: No adapter path provided, using base model.")
    
    model_to_use.eval() # Set final model to evaluation mode
    print(f"INFER: Final model is ready for inference on device: {model_to_use.device if hasattr(model_to_use, 'device') else 'N/A'}.")

    print(f"INFER: Attempting to load tokenizer for {base_model_name}...")
    try:
        # It's generally best to use the tokenizer associated with the base model,
        # unless the fine-tuning process specifically saved a modified tokenizer with the adapters.
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            print("INFER: Tokenizer missing pad_token. Setting pad_token = eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        # Ensure padding side is consistent if it matters for your model/generation
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
# Set ADAPTER_CHECKPOINT_PATH to the directory containing your 'adapter_model.bin' / 'adapter_model.safetensors'
# and 'adapter_config.json'
# ADAPTER_CHECKPOINT_PATH = "./continuoAI_finetuned_thesession_v1_sample/final_checkpoint/"
# To test with ONLY the base model, uncomment the line below and comment out the one above:
ADAPTER_CHECKPOINT_PATH = None 

# If your LoRA adapters were trained on a quantized base model (QLoRA),
# it's generally necessary to also load the base model quantized for inference with those adapters.
USE_QUANTIZATION_FOR_BASE_INFERENCE = True if ADAPTER_CHECKPOINT_PATH else False
# If ADAPTER_CHECKPOINT_PATH is None (using base model), you can set USE_QUANTIZATION_FOR_BASE_INFERENCE
# to True if you still want 4-bit inference for the base model to save VRAM, or False for full float16.
# For base model only, let's default to float16 unless VRAM is an issue:
if ADAPTER_CHECKPOINT_PATH is None:
    USE_QUANTIZATION_FOR_BASE_INFERENCE = False # Change to True if you need to quantize base for VRAM

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
    traceback.print_exc() # Also print traceback for global load errors

def generate_music_continuation(prompt_abc: str,
                                max_new_tokens=128, # Defaulting to shorter for quicker tests
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
            min_new_tokens=10, # Ensure at least some generation
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
    print(f"Testing with adapters from: {ADAPTER_CHECKPOINT_PATH if ADAPTER_CHECKPOINT_PATH else 'BASE MODEL ONLY'}")
    print(f"Base model quantization for inference: {USE_QUANTIZATION_FOR_BASE_INFERENCE}")
    print("="*30 + "\n")

    if MODEL and TOKENIZER:
        # --- Test Case 1: ABC Continuation (Focus on this for your fine-tuned model) ---
        # Use a prompt that might have been similar in structure to your fine-tuning data
        abc_start_snippet_for_test = """X:1
T:Test Completion Start - G Major
M:4/4
L:1/8
K:Gmaj
G A B c | d2 e2""" # Shorter prompt
        
        instruction1 = f"Here is the beginning of a musical piece. Please continue it in a coherent style:\n{abc_start_snippet_for_test}"
        prompt1 = f"Human: {instruction1}</s> Assistant: "

        print(f"\n--- Test 1: ABC Continuation (G Major) ---")
        print(f"Prompting with:\n{prompt1}")
        continuation1 = generate_music_continuation(prompt1, max_new_tokens=128) 
        print("\nGenerated Music (ABC Continuation):")
        print(continuation1)
        print("--------------------------------")
        
        # --- Test Case 2: Another ABC Continuation example ---
        abc_start_snippet_for_test_2 = """X:1
T:Another Test - D Major
M:6/8
L:1/8
K:Dmaj
A | dcd FGA | Bcd"""
        instruction2 = f"Complete the following musical phrase:\n{abc_start_snippet_for_test_2}"
        prompt2 = f"Human: {instruction2}</s> Assistant: "
        print(f"\n--- Test 2: ABC Continuation (D Major) ---")
        print(f"Prompting with:\n{prompt2}")
        continuation2 = generate_music_continuation(prompt2, max_new_tokens=96) # Even shorter for variety
        print("\nGenerated Music (ABC Continuation 2):")
        print(continuation2)
        print("--------------------------------")

    elif MODEL_LOAD_ERROR:
        print(f"Cannot run test: Model or Tokenizer failed to load. Error: {MODEL_LOAD_ERROR}")
    else:
        # This case indicates an issue before MODEL_LOAD_ERROR was even set, likely during import of this script.
        print("Cannot run test: Model or Tokenizer not initialized for an unknown reason (possibly import error in this script itself).")

    print("\n" + "="*30)
    print("inference_engine.py: End of main script execution.")
    print("="*30 + "\n")