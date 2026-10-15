import pandas as pd
import re

df = pd.read_csv('data/raw/reviews.csv')
print("Raw reviews:", len(df))

def label(score):
    if score >= 4:
        return 'positive'
    if score == 3:
        return 'neutral'
    return 'negative'

def clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

df = df.dropna(subset=['content']).drop_duplicates(subset=['content'])
df['sentiment'] = df['score'].apply(label)
df['clean'] = df['content'].apply(clean)
df = df[df['clean'].str.split().str.len() >= 3]

df.to_csv('data/processed/clean_reviews.csv', index=False)
print("Clean reviews:", len(df))
print(df['sentiment'].value_counts())