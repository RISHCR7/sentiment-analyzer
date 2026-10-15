from google_play_scraper import reviews, Sort
import pandas as pd
import time

APP_ID = 'com.application.zomato'
TARGET = 5000
BATCH = 200

all_reviews = []
token = None

while len(all_reviews) < TARGET:
    try:
        result, token = reviews(
            APP_ID, lang='en', country='in',
            sort=Sort.NEWEST, count=BATCH,
            continuation_token=token
        )
    except Exception as e:
        print("Stopped due to error:", e)
        break

    if not result:
        break

    all_reviews.extend(result)
    print("Collected:", len(all_reviews))
    time.sleep(1)

df = pd.DataFrame(all_reviews)[['content', 'score', 'at']]
df.to_csv('data/raw/reviews.csv', index=False)
print("Saved", df.shape, "to data/raw/reviews.csv")