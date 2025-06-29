import requests
import time
import traceback

print("scraper.py: V2 Loaded - Ready for scaled acquisition.")

def get_abc_from_thesession(url):
    """Specific scraper for thesession.org/tunes/ID/abc endpoints."""
    try:
        # Using a browser-like user-agent can sometimes help avoid being blocked
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status() # Raise an exception for HTTP errors (like 500, 503)
        
        # We know these endpoints should be text/plain and start with 'X:'
        abc_text = response.text.strip()
        if abc_text and abc_text.lstrip().startswith("X:"):
            return [abc_text] # Return as a list containing one large string block
        return []
    except requests.exceptions.HTTPError as http_err:
        if hasattr(http_err.response, 'status_code') and http_err.response.status_code == 404:
            # 404 is normal for non-existent tune IDs, so we just pass silently.
            pass
        else:
            print(f"HTTP error for {url}: {http_err}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching {url}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while processing {url}: {e}")
        return []

def save_abc_data(abc_strings_list, filename="scraped_abc_data.txt"):
    """Saves a list of ABC string blocks to a file."""
    tune_separator = "\n\n%-------------------- TUNE SEPARATOR --------------------%\n\n"
    with open(filename, 'w', encoding='utf-8') as f:
        for i, abc_string in enumerate(abc_strings_list):
            f.write(abc_string.strip())
            if i < len(abc_strings_list) - 1:
                f.write(tune_separator)
    print(f"Saved {len(abc_strings_list)} ABC string blocks to {filename}")

if __name__ == "__main__":
    print("Running scraper.py (V2) for scaled data acquisition...")

    # Define a range of tune IDs to scrape from thesession.org
    START_ID = 1
    END_ID = 500  # Scrape first 500 tunes. Adjust this number as needed.
    
    target_urls = [f"https://thesession.org/tunes/{i}/abc" for i in range(START_ID, END_ID + 1)]
    
    print(f"Will attempt to scrape {len(target_urls)} URLs from thesession.org...")

    all_scraped_abc = []
    success_count = 0
    fail_count = 0

    for i, url in enumerate(target_urls):
        if (i + 1) % 50 == 0:
            print(f"Scraping progress: {i+1}/{len(target_urls)}")
            
        abc_data = get_abc_from_thesession(url)
        if abc_data:
            all_scraped_abc.extend(abc_data)
            success_count += 1
        else:
            fail_count += 1
        
        # Be polite to the server: a small delay prevents getting blocked.
        time.sleep(0.1)

    print(f"\nScraping complete. Successfully fetched from {success_count} URLs, failed for {fail_count} URLs.")

    if all_scraped_abc:
        # Save to a new, clearly named file for this version of the data
        save_abc_data(all_scraped_abc, "initial_scraped_tunes_v2_diverse.txt")
    else:
        print("No ABC data was scraped successfully in this test run.")
