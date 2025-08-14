import os
import pandas as pd
import re
from glob import glob
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm 

nltk.download('punkt', quiet=True)

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9\s.,!?'\"]+", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

base_dir = r"C:\Users\secki\OneDrive\Desktop\MY498 Capstone Under Supervision\UNGA Speech Corpus\Jankin Full Dataset\TXT"
speaker_file = r"C:\Users\secki\OneDrive\Desktop\MY498 Capstone Under Supervision\UNGA Speech Corpus\Jankin Full Dataset\Speakers_by_session.xlsx"

valid_years = list(range(2017, 2023))
valid_countries = {
    "AND", "ARG", "AUS", "BGD", "ARM", "BOL", "BRA", "MMR", "CAN", "CHL", "CHN",
    "COL", "CYP", "CZE", "ECU", "ETH", "DEU", "GRC", "GTM", "IND", "IDN", "IRN",
    "IRQ", "JPN", "KAZ", "JOR", "KEN", "KOR", "KGZ", "LBN", "LBY", "MYS", "MDV",
    "MEX", "MNG", "MAR", "NLD", "NZL", "NIC", "NGA", "PAK", "PER", "PHL", "ROU",
    "RUS", "SRB", "SGP", "SVK", "VNM", "ZWE", "TJK", "THA", "TUN", "TUR", "UKR",
    "EGY", "GBR", "USA", "URY", "UZB", "VEN"
}

# LOAD SPEAKER METADATA
speakers = pd.read_excel(speaker_file)
speakers = speakers[["Year", "Session", "ISO Code", "Country", "Name of Person Speaking", "Post"]]
speakers = speakers.rename(columns={
    "Year": "A_YEAR",
    "Session": "A_SESSION",
    "ISO Code": "B_COUNTRY_ALPHA",
    "Name of Person Speaking": "speaker_name",
    "Post": "speaker_post"
})
speakers["B_COUNTRY_ALPHA"] = speakers["B_COUNTRY_ALPHA"].str.upper()

# PARSE SPEECH FILES
all_speech_rows = []

session_dirs = glob(os.path.join(base_dir, "Session * - *"))
for session_dir in tqdm(session_dirs, desc="Sessions"):
    txt_files = glob(os.path.join(session_dir, "*.txt"))
    for path in tqdm(txt_files, desc=f"Files in {os.path.basename(session_dir)}", leave=False):
        fname = os.path.basename(path)
        if fname.startswith("._"):
            continue
        match = re.match(r"([A-Z]{3})_(\d{2})_(\d{4})\.txt", fname)
        if not match:
            continue
        iso, session, year_str = match.groups()
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        all_speech_rows.append({
            "B_COUNTRY_ALPHA": iso,
            "A_SESSION": int(session),
            "A_YEAR": int(year_str),
            "doc": text
        })

df_speeches = pd.DataFrame(all_speech_rows)

# FILTER BY YEAR AND COUNTRY TO KEEP FULL DATASET
#df_speeches = df_speeches[
 #    df_speeches["A_YEAR"].isin(valid_years) &
  #   df_speeches["B_COUNTRY_ALPHA"].isin(valid_countries)
 #].reset_index(drop=True)

# JOIN WITH METADATA
df = df_speeches.merge(speakers, on=["B_COUNTRY_ALPHA", "A_SESSION", "A_YEAR"], how="left")

# CLEAN TEXT
df['doc_clean'] = df['doc'].apply(clean_text)

# CONVERT TO SENTENCE LEVEL WITH doc_id
sentence_rows = []
for _, row in tqdm(df.iterrows(), desc="Docs", total=df.shape[0]):
    doc_id = f"{row['B_COUNTRY_ALPHA']}_{row['A_SESSION']}_{row['A_YEAR']}"
    sentences = sent_tokenize(row['doc_clean'])
    for sent in sentences:
        sentence_rows.append({
            "doc_id": doc_id,
            "B_COUNTRY_ALPHA": row['B_COUNTRY_ALPHA'],
            "A_SESSION": row['A_SESSION'],
            "A_YEAR": row['A_YEAR'],
            "speaker_name": row['speaker_name'],
            "speaker_post": row['speaker_post'],
            "sentence_text": sent
        })

# Filter the sentences between 5 and 50 words
filtered_sentence_rows = [
    r for r in sentence_rows 
    if 10 < len(r['sentence_text'].split()) <= 40
]
df_sentences = pd.DataFrame(filtered_sentence_rows)

# SHUFFLE
df_sentences = df_sentences.sample(frac=1).reset_index(drop=True)

# SAVE TO CSV
output_path = r"C:\Users\secki\OneDrive\Desktop\MY498 Capstone Under Supervision\UNGA Speech Corpus\Jankin Full Dataset\capstone_pipeline\unga_speech_corpus.csv"
df_sentences.to_csv(output_path, index=False, encoding="utf-8")

# PRINT FIRST 5 ROWS
from IPython.display import display
display(df_sentences.head(5))
