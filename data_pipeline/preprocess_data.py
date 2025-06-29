import json
import re
import traceback

print("preprocess_data.py: V2 Loaded.")

def clean_music_line(line):
    """Removes comments and other non-music elements from a line of ABC."""
    line = line.split('%')[0]
    line = re.sub(r'\[I:[^\]]*\]', '', line)
    line = re.sub(r'^w:.*', '', line, flags=re.MULTILINE)
    return line.strip()

def split_abc_tune(single_setting_abc, split_point_bars=4, min_completion_bars=4):
    """
    V2 of the splitting function. More robust cleaning and header detection.
    """
    lines = single_setting_abc.strip().split('\n')
    header_lines = []
    music_lines = []
    
    KNOWN_HEADERS = ('X:', 'T:', 'C:', 'S:', 'Z:', 'O:', 'A:', 'R:', 'M:', 'L:', 'Q:', 'P:', 'K:', 'N:', 'G:', 'H:', 'V:')

    for line in lines:
        cleaned_line = clean_music_line(line)
        if not cleaned_line:
            continue

        if cleaned_line.startswith(KNOWN_HEADERS):
            header_lines.append(cleaned_line)
        else:
            music_lines.append(cleaned_line)
            
    if not any(line.startswith("K:") for line in header_lines) or not music_lines:
        return None, None

    full_music_notation = " ".join(music_lines)
    bars = [b.strip() for b in ' '.join(full_music_notation.split()).split('|')]
    bars = [b for b in bars if b and not b.isspace()]

    if len(bars) < split_point_bars + min_completion_bars:
        return None, None

    prompt_bars = bars[:split_point_bars]
    completion_bars = bars[split_point_bars:]
    
    if not prompt_bars or not completion_bars: return None, None
    
    prompt_abc_full = "\n".join(header_lines) + "\n" + " | ".join(prompt_bars) + " |"
    completion_text = " | ".join(completion_bars).strip()
    
    if completion_text.strip() in (':|', '||', '|]'): return None, None

    return prompt_abc_full, completion_text

def create_finetuning_record(prompt_abc, completion_abc, instruction="Please continue this musical piece in a similar style:"):
    if not prompt_abc or not completion_abc: return None
    prompt_text = f"Human: {instruction}\n{prompt_abc}</s> Assistant: "
    return {"prompt": prompt_text, "completion": completion_abc}

def process_scraped_data(input_filename, output_jsonl_filename, max_examples=5000):
    """V2: Reads scraped ABC data, splits multi-setting blocks, and creates a JSONL dataset."""
    fetched_block_separator = "%-------------------- TUNE SEPARATOR --------------------%"
    records_created = 0

    try:
        print(f"Reading from input file: {input_filename}")
        with open(input_filename, 'r', encoding='utf-8') as infile:
            full_content = infile.read()
        
        fetched_blocks = full_content.split(fetched_block_separator)
        print(f"Found {len(fetched_blocks)} fetched blocks.")

        all_individual_settings = []
        for tune_block_raw in fetched_blocks:
            if not tune_block_raw.strip(): continue
            settings_in_block = re.split(r'\n\s*\n(?=X:)', tune_block_raw.strip())
            all_individual_settings.extend(settings_in_block)

        print(f"Total individual tune settings found: {len(all_individual_settings)}")
        
        print(f"Processing settings and writing to: {output_jsonl_filename}")
        with open(output_jsonl_filename, 'w', encoding='utf-8') as outfile:
            for setting_abc in all_individual_settings:
                if records_created >= max_examples:
                    print(f"\nReached max_examples limit of {max_examples}.")
                    break
                
                prompt_part, completion_part = split_abc_tune(setting_abc)

                if prompt_part and completion_part:
                    record = create_finetuning_record(prompt_part, completion_part)
                    if record:
                        outfile.write(json.dumps(record) + '\n')
                        records_created += 1
                        if records_created % 500 == 0:
                            print(f"Progress: {records_created} records created...")
            
    except FileNotFoundError:
        print(f"Error: Input file {input_filename} not found. Please run the scraper first.")
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        traceback.print_exc()

    print(f"\nFinished preprocessing. Created {records_created} records in {output_jsonl_filename}.")

if __name__ == "__main__":
    print("Running preprocess_data.py (V2) on new large scraped file...")
    process_scraped_data(
        input_filename="initial_scraped_tunes_v2_diverse.txt",
        output_jsonl_filename="finetuning_dataset_v2_refined.jsonl",
        max_examples=5000  # You can adjust this limit
    )
