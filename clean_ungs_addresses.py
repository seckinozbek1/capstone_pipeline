import os
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm
from IPython.display import display


nltk.download('punkt', quiet=True)

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9\s.,!?'\"]+", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Directory where the speech files are stored
base_dir = r"ungs_speeches"

# Collect sentences from all files
sentence_rows = []

# List all text files matching "ungs_*.txt"
files = [f for f in os.listdir(base_dir) if re.match(r"ungs_\d{4}\.txt", f)]

for filename in tqdm(files, desc="Processing files"):
    filepath = os.path.join(base_dir, filename)
    # Extract year from filename, e.g., "ungs_2017.txt" -> 2017
    year_match = re.search(r"ungs_(\d{4})\.txt", filename)
    if not year_match:
        continue
    year = int(year_match.group(1))
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    text_clean = clean_text(text)
    sentences = sent_tokenize(text_clean)
    
    # Filter sentences between 10 and 40 words (inclusive)
    for sent in sentences:
        word_count = len(sent.split())
        if 10 < word_count <= 40:
            sentence_rows.append({
                "doc_id": f"ungs_{year}",
                "A_YEAR": year,
                "sentence_text": sent
            })

# Create DataFrame
df_sentences = pd.DataFrame(sentence_rows)

# Shuffle the dataframe rows
df_sentences = df_sentences.sample(frac=1).reset_index(drop=True)

# Save to CSV
output_path = r"C:\Users\secki\OneDrive\Desktop\MY498 Capstone Under Supervision\UNGA Speech Corpus\Jankin Full Dataset\capstone_pipeline\ungs_address_corpus.csv"
df_sentences.to_csv(output_path, index=False, encoding="utf-8")

print(f"Processed {len(df_sentences)} sentences from {len(files)} files.")
pd.set_option('display.max_colwidth', None)
display(df_sentences.head(5))
