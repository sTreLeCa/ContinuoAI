import requests
from bs4 import BeautifulSoup
import time
import traceback

print("scraper.py: Loaded.")

def get_abc_from_url(url):
    """Fetches a page. If it's a known direct /abc endpoint, returns the text."""
    # This function is already quite robust from our previous work.
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if url.endswith("/abc"):
            abc_text = response.text.strip()
            if abc_text and abc_text.lstrip().startswith("X:"):
                return [abc_text]
        return []
    except requests.exceptions.HTTPError as http_err:
        if hasattr(http_err.response, 'status_code') and http_err.response.status_code == 404:
            pass # Suppress 404 errors for mass scraping, it's expected some tunes don't exist.
        else:
            print(f"HTTP error fetching {url}: {http_err}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching {url}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while processing {url}: {e}")
        return []

def save_abc_data(abc_strings_list, filename="scraped_abc_data_large.txt"):
    """Saves a list of ABC strings to a file, each on a new block separated by a standard marker."""
    tune_separator = "\n\n%-------------------- TUNE SEPARATOR --------------------%\n\n"
    with open(filename, 'w', encoding='utf-8') as f: # Use 'w' to overwrite the file each time this script runs
        for i, abc_string in enumerate(abc_strings_list):
            f.write(abc_string.strip())
            if i < len(abc_strings_list) - 1: # Don't add separator after the last item
                f.write(tune_separator)
    print(f"Saved {len(abc_strings_list)} ABC string blocks to {filename}")


if __name__ == "__main__":
    print("Running scraper.py for scaled data acquisition...")

    # Define a range of tune IDs to scrape from thesession.org
    START_ID = 1
    END_ID = 500  # Let's aim for the first 500 tunes
    
    # Generate the list of target URLs
    target_urls = [f"https://thesession.org/tunes/{i}/abc" for i in range(START_ID, END_ID + 1)]
    
    print(f"Will attempt to scrape {len(target_urls)} URLs from thesession.org...")

    all_scraped_abc = []
    success_count = 0
    fail_count = 0

    for i, url in enumerate(target_urls):
        # The print statement can be verbose, so we'll print progress every 25 tunes
        if (i + 1) % 25 == 0:
            print(f"Scraping progress: {i+1}/{len(target_urls)}")
            
        abc_data_list = get_abc_from_url(url)
        if abc_data_list:
            all_scraped_abc.extend(abc_data_list)
            success_count += 1
        else:
            fail_count += 1
        
        # Be polite to the server: a small delay between requests
        time.sleep(0.2)

    print(f"\nScraping complete. Successfully fetched from {success_count} URLs, failed for {fail_count} URLs.")

    if all_scraped_abc:
        save_abc_data(all_scraped_abc, "initial_scraped_tunes_large_v1.txt")
    else:
        print("No ABC data was scraped successfully in this test run.")
