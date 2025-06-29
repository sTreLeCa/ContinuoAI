import json
import re # For splitting by X:
import traceback

print("preprocess_data.py: Loaded.")

def split_abc_tune(single_setting_abc, split_point_bars=4, min_completion_bars=4):
    """
    Takes a SINGLE ABC tune/setting and splits it into a prompt/completion pair.
    This version has stricter checks.
    """
    lines = single_setting_abc.strip().split('\n')
    header_lines = []
    music_lines = []
    
    # Separate header (lines before K:) and music (lines from K: onwards)
    try:
        k_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("K:"):
                k_index = i
                break
        if k_index == -1: return None, None # No key signature, invalid ABC for our purpose

        header_lines = lines[:k_index+1]
        music_lines = lines[k_index+1:]
    except Exception:
        return None, None
    
    if not music_lines: return None, None

    full_music_notation = " ".join(music_lines)
    bars = [b.strip() for b in ' '.join(full_music_notation.split()).split('|')]
    bars = [b for b in bars if b and not b.isspace()] # Filter out empty/whitespace bars

    if len(bars) < split_point_bars + min_completion_bars:
        return None, None # Tune is too short

    prompt_bars = bars[:split_point_bars]
    completion_bars = bars[split_point_bars:]
    
    if not prompt_bars or not completion_bars: return None, None
    
    prompt_abc_full = "\n".join(header_lines) + "\n" + " | ".join(prompt_bars) + " |"
    completion_text = " | ".join(completion_bars).strip()
    
    # Final check: if completion is just an end marker, discard it
    if completion_text in (':|', '||', '|]'): return None, None

    return prompt_abc_full, completion_text

def create_finetuning_record(prompt_abc, completion_abc, instruction="Please continue this musical piece in a similar style:"):
    if not prompt_abc or not completion_abc: return None
    prompt_text = f"Human: {instruction}\n{prompt_abc}</s> Assistant: "
    return {"prompt": prompt_text, "completion": completion_abc}

def process_scraped_data(input_filename, output_jsonl_filename, max_examples=10000):
    """Reads scraped ABC data, splits multi-setting blocks, and creates a JSONL dataset."""
    fetched_block_separator = "%-------------------- TUNE SEPARATOR --------------------%"
    records_created = 0

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            full_content = infile.read()
        
        # Split into blocks from different URLs
        fetched_blocks = full_content.split(fetched_block_separator)
        print(f"Found {len(fetched_blocks)} fetched blocks in {input_filename}.")

        all_individual_settings = []
        for tune_block_raw in fetched_blocks:
            if not tune_block_raw.strip(): continue
            # Split each block by the X: header to get individual settings/versions
            # The regex splits on a newline that is FOLLOWED BY "X:", keeping the "X:"
            settings_in_block = re.split(r'\n(?=X:)', tune_block_raw.strip())
            all_individual_settings.extend(settings_in_block)

        print(f"Total individual tune settings found: {len(all_individual_settings)}")
        
        with open(output_jsonl_filename, 'w', encoding='utf-8') as outfile:
            for setting_abc in all_individual_settings:
                if records_created >= max_examples:
                    print(f"Reached max_examples limit of {max_examples}.")
                    break
                
                prompt_part, completion_part = split_abc_tune(setting_abc)

                if prompt_part and completion_part:
                    record = create_finetuning_record(prompt_part, completion_part)
                    if record:
                        outfile.write(json.dumps(record) + '\n')
                        records_created += 1
                        if records_created % 100 == 0:
                            print(f"Progress: {records_created} records created...")
            
    except FileNotFoundError:
        print(f"Error: Input file {input_filename} not found.")
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        traceback.print_exc()

    print(f"\nFinished preprocessing. Created {records_created} records in {output_jsonl_filename}.")

if __name__ == "__main__":
    print("Running preprocess_data.py for scaled data acquisition...")
    process_scraped_data(
        input_filename="initial_scraped_tunes_large_v1.txt",
        output_jsonl_filename="finetuning_dataset_large_v1.jsonl"
    )
