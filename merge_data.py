import pandas as pd
import re

# --- load your existing project dataset (label,text) ---
existing = pd.read_csv('data/spam.csv', encoding='latin-1', header=None, names=['label', 'text'])

# drop a stray header row if spam.csv already has one at the top
existing = existing[~existing['label'].str.lower().isin(['label', 'v1'])]

# --- load the second dataset ---
second = pd.read_csv('data/emails.csv', sep=None, engine='python', header=None)

# drop fully empty trailing columns (handles the extra trailing tab you pasted)
second = second.dropna(axis=1, how='all')

# find the 0/1 label column and the text column automatically
label_col = None
for c in second.columns:
    vals = pd.to_numeric(second[c], errors='coerce').dropna().unique()
    if set(vals).issubset({0, 1}) and len(vals) > 0:
        label_col = c
        break

text_col = [c for c in second.columns if c != label_col][0]

second = second.rename(columns={text_col: 'text', label_col: 'raw_label'})
second['raw_label'] = pd.to_numeric(second['raw_label'], errors='coerce')
second = second.dropna(subset=['raw_label'])

# --- strip the leading "Subject:" from every message ---
second['text'] = second['text'].astype(str).str.replace(
    r'^\s*Subject\s*:\s*', '', regex=True, flags=re.IGNORECASE
).str.strip()

# --- all spam rows ---
spam_rows = second[second['raw_label'] == 1].copy()
spam_rows['label'] = 'spam'

# --- 200 random ham rows ---
ham_rows = second[second['raw_label'] == 0].sample(n=200, random_state=42).copy()
ham_rows['label'] = 'ham'

to_add = pd.concat([spam_rows, ham_rows])[['label', 'text']]

# --- append to the existing project dataset and save ---
combined = pd.concat([existing, to_add], ignore_index=True)
combined.to_csv('data/spam.csv', index=False, header=False, encoding='latin-1')

print(f"Added {len(spam_rows)} spam rows and {len(ham_rows)} ham rows.")
print(f"New total in data/spam.csv: {len(combined)} rows.")