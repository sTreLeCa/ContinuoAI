import requests
from bs4 import BeautifulSoup
import time

print("scraper.py: Loaded.")

def get_abc_from_url(url):
    """Fetches a page. If it's a known direct /abc endpoint, returns the text. Otherwise, tries basic HTML parsing."""
    print(f"Attempting to fetch ABC from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Check if the URL itself indicates it's a raw ABC endpoint (more reliable for thesession.org)
        if url.endswith("/abc"): # Specific check for thesession.org /abc pattern
            print(f"URL ends with /abc, assuming direct ABC content for {url}.")
            abc_text = response.text.strip()
            if abc_text:
                # Sometimes these /abc endpoints might still have a bit of HTML or other text if an error occurs on their side
                # A simple check: ABC usually starts with X:
                if abc_text.lstrip().startswith("X:"):
                    print(f"Content from {url} starts with 'X:', confirmed as ABC.")
                    return [abc_text]
                else:
                    print(f"Content from {url} does not start with 'X:'. It might not be raw ABC. Content preview: {abc_text[:200]}")
                    # Try basic HTML parsing as a fallback if it wasn't plain text but was an /abc URL
                    # This part is less likely to be hit for valid /abc URLs from thesession
                    soup = BeautifulSoup(response.content, 'html.parser')
                    pre_tags = soup.find_all('pre')
                    if pre_tags:
                        abc_texts = [pre.get_text(separator='\n', strip=True) for pre in pre_tags if pre.get_text(strip=True).lstrip().startswith("X:")]
                        if abc_texts:
                            print(f"Found {len(abc_texts)} ABC string(s) via fallback HTML 'pre' tag parsing on {url}")
                            return abc_texts
                    print(f"No valid ABC found even with fallback parsing for /abc URL: {url}")
                    return []
            else:
                print(f"Empty content from direct /abc pattern URL: {url}")
                return []
        else:
            # Fallback to BeautifulSoup for pages that might embed ABC in HTML (for other, non-/abc URL sites)
            print(f"URL does not end with /abc. Attempting general HTML parsing for {url}.")
            soup = BeautifulSoup(response.content, 'html.parser')
            # This is a generic placeholder - WILL LIKELY NEED ADJUSTMENT FOR OTHER SITES
            abc_elements = soup.find_all('pre', class_='abc-notation')
            if not abc_elements:
                abc_elements = soup.find_all('div', id='abc-content')
            if not abc_elements: # Broader search if specific classes fail
                 abc_elements = soup.find_all('pre')


            abc_texts = [element.get_text(separator='\n', strip=True) for element in abc_elements if element.get_text(strip=True).lstrip().startswith("X:")]

            if abc_texts:
                print(f"Found {len(abc_texts)} ABC string(s) via HTML parsing on {url}")
                return abc_texts
            else:
                print(f"No ABC notation found via HTML parsing on {url} with current selectors.")
                return []

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404: # Check if response object exists
            print(f"Error 404: Tune not found at {url}")
        else:
            print(f"HTTP error fetching {url}: {http_err}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching {url}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while processing {url}: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_abc_data(abc_strings_list, filename="scraped_abc_data.txt"):
    tune_separator = "\n\n%-------------------- TUNE SEPARATOR --------------------%\n\n"
    with open(filename, 'w', encoding='utf-8') as f:
        for i, abc_string in enumerate(abc_strings_list):
            f.write(abc_string.strip())
            if i < len(abc_strings_list) - 1:
                f.write(tune_separator)
    print(f"Saved {len(abc_strings_list)} ABC string(s) to {filename}")


if __name__ == "__main__":
    print("Running scraper.py as main script for testing...")
    target_urls = [
        "https://thesession.org/tunes/1/abc",
        "https://thesession.org/tunes/7/abc",
        "https://thesession.org/tunes/9999999/abc" # Non-existent
    ]
    all_scraped_abc = []
    for url in target_urls:
        time.sleep(1)
        abc_data_list = get_abc_from_url(url)
        if abc_data_list: # Ensures abc_data_list is not None and not empty
            all_scraped_abc.extend(abc_data_list)

    if all_scraped_abc:
        save_abc_data(all_scraped_abc, "initial_scraped_tunes_v1.txt")
    else:
        print("No ABC data was scraped successfully in this test run.")
