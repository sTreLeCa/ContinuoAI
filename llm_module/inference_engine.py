import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import traceback # For detailed error printing

print("inference_engine.py: Top-level imports loaded.")

def load_chat_musician_model(model_name="m-a-p/ChatMusician"):
    """Loads the ChatMusician model and tokenizer from Hugging Face."""
    print(f"load_chat_musician_model: Attempting to load tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"load_chat_musician_model: Tokenizer for {model_name} loaded successfully.")
    except Exception as e:
        print(f"load_chat_musician_model: ERROR - Failed to load tokenizer for {model_name}. Exception: {e}")
        traceback.print_exc()
        raise

    print(f"load_chat_musician_model: Attempting to load model {model_name}...")
    print("load_chat_musician_model: This might take a while and download several GBs if it's the first time.")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            resume_download=True
        ).eval()  # Set to evaluation mode immediately after loading
        print(f"load_chat_musician_model: Model {model_name} loaded successfully to device: {model.device}.")
    except Exception as e:
        print(f"load_chat_musician_model: ERROR - Failed to load model {model_name}. Exception: {e}")
        traceback.print_exc()
        raise

    return model, tokenizer

# --- Global Model and Tokenizer ---
MODEL = None
TOKENIZER = None
MODEL_LOAD_ERROR = None

print("inference_engine.py: Attempting to load global MODEL and TOKENIZER...")
try:
    MODEL, TOKENIZER = load_chat_musician_model()
    print("inference_engine.py: Global MODEL and TOKENIZER loaded successfully.")
except Exception as e:
    MODEL_LOAD_ERROR = str(e) # Store the error message
    print(f"inference_engine.py: CRITICAL ERROR during global model load: {MODEL_LOAD_ERROR}")
    # MODEL and TOKENIZER will remain None, caught in generate_music_continuation and __main__

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
        error_message = f"generate_music_continuation: Model or Tokenizer not loaded. Stored load error (if any): {MODEL_LOAD_ERROR}"
        print(error_message)
        return error_message # Return error message instead of raising here

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

# This block will only execute when the script is run directly (e.g., python llm_module/inference_engine.py)
if __name__ == "__main__":
    print("\n" + "="*30)
    print("Running inference_engine.py as main script for testing...")
    print("="*30 + "\n")

    if MODEL and TOKENIZER:
        # --- Test Case 1: Chord Progression ---
        instruction1 = """Develop a musical piece using the given chord progression.
'Dm', 'C', 'Dm', 'Dm', 'C', 'Dm', 'C', 'Dm'""" # Keep multi-line for readability
        prompt1 = f"Human: {instruction1}</s> Assistant: "

        print(f"\n--- Test 1: Chord Progression ---")
        print(f"Prompting with:\n{prompt1}")
        continuation1 = generate_music_continuation(prompt1, max_new_tokens=350)
        print("\nGenerated Music (Chord Progression):")
        print(continuation1)
        print("---------------------------------")

        # --- Test Case 2: ABC Continuation ---
        abc_start_snippet = """X:1
T:My Tune to Continue
M:4/4
L:1/8
K:Gmaj
G A B c | d2 e2 d c | B G A G | E2 D2"""
        instruction2 = f"Here is the beginning of a musical piece. Please continue it in a coherent style:\n{abc_start_snippet}"
        prompt2 = f"Human: {instruction2}</s> Assistant: "

        print(f"\n--- Test 2: ABC Continuation ---")
        print(f"Prompting with:\n{prompt2}")
        continuation2 = generate_music_continuation(prompt2, max_new_tokens=256)
        print("\nGenerated Music (ABC Continuation):")
        print(continuation2)
        print("--------------------------------")

        # --- Test Case 3: Text-to-Music (Simple Description) ---
        instruction3 = "Develop a short, happy tune in the key of C major suitable for a children's game."
        prompt3 = f"Human: {instruction3}</s> Assistant: "

        print(f"\n--- Test 3: Text-to-Music ---")
        print(f"Prompting with:\n{prompt3}")
        continuation3 = generate_music_continuation(prompt3, max_new_tokens=200)
        print("\nGenerated Music (Text-to-Music):")
        print(continuation3)
        print("-----------------------------")

    elif MODEL_LOAD_ERROR:
        print(f"Cannot run tests: Model or Tokenizer failed to load during module initialization. Error: {MODEL_LOAD_ERROR}")
    else:
        print("Cannot run tests: Model or Tokenizer not initialized. Check loading logic and console output for errors during module import.")

    print("\n" + "="*30)
    print("inference_engine.py: End of main script execution.")
    print("="*30 + "\n")