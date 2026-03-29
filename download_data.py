import os
import requests

os.makedirs('data/raw', exist_ok=True)
url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"

print("Downloading dataset...")
response = requests.get(url)
with open('data/raw/wikitext-2.txt', 'w', encoding='utf-8') as f:
    # Clean up empty lines and headers
    lines = [line.strip() for line in response.text.split('\n') if len(line.strip()) > 0 and not line.startswith(' = ')]
    f.write('\n'.join(lines))
print("Done! Saved to data/raw/wikitext-2.txt")