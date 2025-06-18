import gradio as gr
import sys

# Ensure the llm_module can be found
if '.' not in sys.path:
    sys.path.append('.')

# This import will use the already loaded MODEL and TOKENIZER
# from the inference_engine module
from llm_module.inference_engine import generate_music_continuation as llm_generate_continuation

print("app.py: Imports loaded. LLM model and tokenizer should be loaded.")

def generate_for_gradio(abc_start_text):
    print(f"Gradio received for generation: {abc_start_text}")
    if not abc_start_text or abc_start_text.strip() == "":
        return "Please enter some ABC music to start."

    prompt = f"""Human: Here is the beginning of a musical piece. Please continue it in a coherent style:
{abc_start_text}</s> Assistant: """

    print(f"Gradio constructed prompt (first 100 chars): {prompt[:100]}")
    continuation = llm_generate_continuation(prompt)
    print(f"Gradio received continuation from LLM: {continuation}")
    return continuation

iface = gr.Interface(
    fn=generate_for_gradio,
    inputs=gr.Textbox(lines=10, placeholder="Enter ABC music start here...\ne.g.,\nX:1\nT:My Tune\nM:4/4\nL:1/8\nK:Gmaj\nG A B c | d2 e2 d c", label="ABC Music Start"),
    outputs=gr.Textbox(lines=10, label="Generated Continuation"),
    title="ContinuoAI: LLM-Powered Music Completion",
    description="Enter the beginning of a piece in ABC notation. The LLM will attempt to generate a continuation using the base m-a-p/ChatMusician model."
)

if __name__ == '__main__':
    print("app.py: Launching Gradio interface...")
    iface.queue().launch(share=True, debug=True)
