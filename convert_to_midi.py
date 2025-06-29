# convert_to_midi.py

from symusic import Score
import traceback

def save_abc_as_midi(abc_text: str, output_filename: str = "output.mid"):
    """
    Takes a string of ABC notation and saves it as a MIDI file.
    """
    if not abc_text or not abc_text.strip():
        print("save_abc_as_midi: No ABC text provided.")
        return

    print("\n--- Converting ABC to MIDI ---")
    print(f"ABC Input (first 100 chars):\n{abc_text[:100]}...")
    
    try:
        # 1. Parse the ABC text string into a symusic Score object
        score = Score.from_abc(abc_text)
        
        # 2. Dump the score object to a MIDI file
        # The 'path' argument specifies where to save the file.
        score.dump_midi(path=output_filename)

        print(f"SUCCESS: Successfully saved music to '{output_filename}'")

    except Exception as e:
        print(f"\nERROR: Could not convert the ABC string to MIDI.")
        print(f"It might contain invalid notation that the parser cannot handle.")
        print(f"Error details: {e}\n")
        traceback.print_exc()

if __name__ == '__main__':
    # --- Example 1: A simple, known-good ABC tune ---
    print("\n--- Converting Example 1: Simple Scale ---")
    simple_scale = """
X:1
T:Simple C Major Scale
M:4/4
L:1/4
K:C
C D E F | G A B c ||
"""
    save_abc_as_midi(simple_scale, "example1_scale.mid")


    # --- Example 2: Your fine-tuned model's successful output ---
    print("\n--- Converting Example 2: Your Fine-Tuned Model's Output ---")
    fine_tuned_output = """
X:1
T:Fine-Tuned G Major Continuation
L:1/8
Q:1/4=96
M:4/4
K:G
G A B c | d2 e2 | GABc dBce | dBGB A2 z D | GABc dBce | dBGB A3 D |
GABc dBce | dBGB A2 z G | GABc dBce | dBGB A2 z2 | B2 AG ABcd |
e2 ed ef g2 | B2 AG ABcd | edef g4 | B2 AG ABcd | e2 ed ef g2 ||
"""
    save_abc_as_midi(fine_tuned_output, "example2_finetuned_music.mid")

    # --- Example 3: The other output from your fine-tuned model ---
    # Note: I'm removing the leading "3.000..." as it will cause a parsing error.
    # This highlights that some minor cleanup of the LLM's output might be needed.
    print("\n--- Converting Example 3: Another Fine-Tuned Output (Cleaned) ---")
    fine_tuned_output_2_cleaned = """
X:1
T:Fine-Tuned D Major Continuation (Cleaned)
M:6/8
L:1/8
K:Dmaj
dcd FGA Bcd | A2 d efg | fed cBA |
"""
    save_abc_as_midi(fine_tuned_output_2_cleaned, "example3_finetuned_music_cleaned.mid")