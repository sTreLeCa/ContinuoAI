import json
import random # For potential random splitting
import traceback # For more detailed error logging

print("preprocess_data.py: Loaded.")

def split_abc_tune(full_abc_text, split_point_bars=4, min_completion_bars=2):
    """
    Splits a full ABC tune into a prompt (start) and completion (rest).
    Assumes bars are separated by '|'.
    """
    # CORRECTED NEWLINE SPLITTING HERE:
    lines = full_abc_text.strip().split('\n') # Use actual newline '\n'
    header_lines = []
    music_lines_text = []

    # Heuristic to separate header from music lines
    # This can be improved, e.g. by looking for the K: field as the end of typical headers
    key_field_found = False
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped: # skip empty lines
            continue
        
        # A common pattern: headers end when K: is found or music notation starts
        if not key_field_found:
            if line_stripped.startswith("K:"):
                header_lines.append(line_stripped)
                key_field_found = True # From here on, assume music lines
            elif line_stripped.startswith(('X:', 'T:', 'M:', 'L:', 'R:', 'C:', 'S:', 'Z:', 'N:', 'O:', 'A:', 'B:', 'Q:', 'P:')):
                header_lines.append(line_stripped)
            elif key_field_found: # If K: was already found, subsequent lines are music
                music_lines_text.append(line_stripped)
            else: # Before K: if it's not a recognized header, might be music or other info
                  # For now, let's assume lines not matching typical headers before K: are part of the music section
                  # or an unhandled header type. A more robust parser would handle this better.
                  # A simple check: if it contains '|' or notes (a-g, A-G) it's likely music.
                if '|' in line_stripped or any(c in "abcdefgABCDEFG" for c in line_stripped):
                    music_lines_text.append(line_stripped)
                else: # Otherwise, assume it's part of the header block
                    header_lines.append(line_stripped)
        else: # After K: field, all non-empty lines are music
            music_lines_text.append(line_stripped)

    if not header_lines or not music_lines_text:
        print(f"Debug: Could not separate header/music for tune. Header: {len(header_lines)}, Music: {len(music_lines_text)}. Tune: {full_abc_text[:100]}...")
        return None, None
    if not key_field_found and "K:" not in "\n".join(header_lines): # Double check K: presence
        print(f"Warning: No K: field clearly identified in header for tune starting with {header_lines[0] if header_lines else 'N/A'}")
        # Depending on strictness, you might return None, None here

    # Process music lines
    full_music_notation = " ".join(music_lines_text)
    # Normalize multiple spaces that might result from joining lines, then split by bar
    bars = [b.strip() for b in ' '.join(full_music_notation.split()).split('|')]
    bars = [b for b in bars if b] # Remove empty strings

    if len(bars) < split_point_bars + min_completion_bars:
        print(f"Debug: Tune too short. Total bars: {len(bars)}, Required: {split_point_bars + min_completion_bars}. Tune: {header_lines[0] if header_lines else 'N/A'}")
        return None, None

    prompt_bars_list = bars[:split_point_bars]
    completion_bars_list = bars[split_point_bars:]

    if not prompt_bars_list or not completion_bars_list:
        print(f"Debug: Prompt or completion bars list is empty after split. Tune: {header_lines[0] if header_lines else 'N/A'}")
        return None, None

    prompt_abc_music = " | ".join(prompt_bars_list) + " |"
    prompt_abc_full = "\n".join(header_lines) + "\n" + prompt_abc_music

    completion_abc_music = " | ".join(completion_bars_list)
    # Basic check to ensure completion isn't just an end bar like '||' or ':|'
    if completion_abc_music.strip() in ('||', ':|', '|]', '|'):
        print(f"Debug: Completion part is only an end bar symbol. Tune: {header_lines[0] if header_lines else 'N/A'}")
        return None, None # Completion is too short or just an end marker

    # Ensure completion ends correctly if it's the end of a piece (already in your code, good)
    # This might need adjustment based on how the source ABC terminates tunes.
    # For now, if it's not a double bar or repeat, we add a single bar.
    # It's better if the source data has proper termination.
    cleaned_completion = completion_abc_music.strip()
    if not cleaned_completion.endswith(('||', ':|', '|]')):
         if cleaned_completion and not cleaned_completion.endswith('|'):
             cleaned_completion += " |"
         elif not cleaned_completion: # if completion is empty after stripping, it's problematic
            print(f"Debug: Completion part is empty after stripping. Tune: {header_lines[0] if header_lines else 'N/A'}")
            return None, None


    # Final basic validation placeholder
    # Could add: check for balanced repeats, valid characters, etc.
    # For now, a simple check that K: is in the prompt part.
    if "K:" not in prompt_abc_full:
        print(f"Warning: No K: field in prompt part. Prompt: {prompt_abc_full[:100]}...")
        # return None, None # Decide if this is a fatal error for V1

    return prompt_abc_full.strip(), cleaned_completion.strip()


# --- Keep create_finetuning_record and process_scraped_data as they were in the previous correct version ---
# --- (or as they are in your Cell 10 if they were already good) ---
def create_finetuning_record(prompt_abc, completion_abc, instruction="Please continue this musical piece in a similar style:"):
    if not prompt_abc or not completion_abc:
        return None
    prompt_text = f"Human: {instruction}\n{prompt_abc}</s> Assistant: "
    record = {
        "prompt": prompt_text,
        "completion": completion_abc
    }
    return record

def process_scraped_data(input_filename="initial_scraped_tunes_v1.txt",
                         output_jsonl_filename="finetuning_dataset_v1.jsonl", # Changed default
                         max_examples=50):
    tune_separator = "%-------------------- TUNE SEPARATOR --------------------%"
    records_created = 0
    number_of_tunes_found = 0

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            full_content = infile.read()
        
        individual_tunes_raw = full_content.split(tune_separator)
        individual_tunes = [tune.strip() for tune in individual_tunes_raw if tune.strip()]
        number_of_tunes_found = len(individual_tunes)
        
        print(f"Found {number_of_tunes_found} actual tunes in {input_filename} after splitting by separator and stripping.")

        with open(output_jsonl_filename, 'w', encoding='utf-8') as outfile:
            for full_abc_tune in individual_tunes:
                if records_created >= max_examples:
                    print(f"Reached max_examples limit of {max_examples}.")
                    break
                
                print(f"\nProcessing tune: {full_abc_tune[: min(100, len(full_abc_tune))]}...")
                prompt_part, completion_part = split_abc_tune(full_abc_tune)

                if prompt_part and completion_part:
                    record = create_finetuning_record(prompt_part, completion_part)
                    if record:
                        # Corrected way to write JSONL with a proper newline
                        outfile.write(json.dumps(record) + '\n') 
                        # OR, preferred:
                        # print(json.dumps(record), file=outfile) 
                        
                        records_created += 1
                        print(f"SUCCESS: Created record {records_created}/{max_examples}")
                        if records_created < 5: 
                            print(json.dumps(record, indent=2)) # For pretty printing to console
                else:
                    print(f"SKIPPED tune due to splitting issue or insufficient length. Tune preview: {full_abc_tune.strip()[:100]}...")
    
    except FileNotFoundError:
        print(f"Error: Input file {input_filename} not found. Scraper might not have run successfully.")
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        traceback.print_exc()

    print(f"\nFinished preprocessing. Created {records_created} records in {output_jsonl_filename}.")
    # ... (rest of the warnings) ...

if __name__ == "__main__":
    print("Running preprocess_data.py as main script for testing...")
    # Ensure the output filename matches what fine_tune.py expects
    process_scraped_data(input_filename="initial_scraped_tunes_v1.txt",
                         output_jsonl_filename="finetuning_dataset_v1_sample.jsonl", 
                         max_examples=50)