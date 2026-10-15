# 💬 Customer Review Sentiment Analyzer

An end-to-end machine learning project that classifies customer app reviews as **positive**, **neutral** or **negative**. It covers the full lifecycle: data collection, cleaning, exploratory analysis, model training and tuning, evaluation, and deployment as a live web app.

**🔗 Live demo:** https://sentiment-analyzer-lvrbkocvka2c7rreojivwa.streamlit.app/

> Built as an MCA end-to-end ML project.

---

## Table of contents

1. [Problem statement](#problem-statement)
2. [Features](#features)
3. [Tech stack](#tech-stack)
4. [Project structure](#project-structure)
5. [Project pipeline](#project-pipeline)
6. [Results](#results)
7. [How to run locally](#how-to-run-locally)
8. [Using the app](#using-the-app)
9. [Limitations](#limitations)
10. [Future scope](#future-scope)
11. [Author](#author)

---

## Problem statement

Businesses receive thousands of customer reviews and cannot read them all. This project automatically classifies each review's sentiment so that companies can quickly see what customers feel and spot problems early.

- **Task:** 3-class text classification (positive / neutral / negative)
- **Input:** review text
- **Output:** sentiment label with a confidence score
- **Metrics:** accuracy and macro F1-score (macro F1 is used because the classes are imbalanced)

---

## Features

- **Single review analysis:** paste a review and get the sentiment, a confidence score and a class probability chart
- **Bulk CSV analysis:** upload a CSV of reviews and get a sentiment summary, charts, a filterable table and downloadable results
- **Model info panel:** shows the model name and test metrics in the sidebar
- **Example reviews** for a quick demo

---

## Tech stack

| Area | Tools |
|---|---|
| Language | Python |
| Data collection | `google-play-scraper` |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly, WordCloud |
| Machine learning | Scikit-learn (TF-IDF, Logistic Regression, LinearSVC, Complement Naive Bayes) |
| Web app | Streamlit |
| Deployment | Streamlit Community Cloud, GitHub |
| Editor | VS Code, Jupyter Notebook |

---

## Project structure

```
Customer Review Sentiment Analyzer/
├── app/
│   └── app.py                    # Streamlit web app
├── data/
│   ├── raw/reviews.csv           # scraped reviews
│   └── processed/clean_reviews.csv
├── models/
│   ├── sentiment_pipeline.joblib # trained TF-IDF + classifier pipeline
│   └── metrics.json              # final test metrics
├── notebooks/
│   └── EDA.ipynb                 # exploratory data analysis
├── reports/
│   ├── figures/                  # saved charts (EDA, confusion matrix, etc.)
│   ├── model_comparison.csv
│   └── misclassified_reviews.csv
├── src/
│   ├── collect.py                # data collection
│   ├── preprocess.py             # cleaning and labeling
│   └── train.py                  # training, tuning, evaluation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Project pipeline

### 1. Data collection (`src/collect.py`)
Customer reviews of the **Zomato** app were scraped from the Google Play Store using `google-play-scraper`, in batches of 200 with a short delay between requests. Each record contains the review text, star rating and date.

### 2. Labeling and cleaning (`src/preprocess.py`)
- **Labels** come from star ratings (weak supervision):
  - 4 to 5 stars: positive
  - 3 stars: neutral
  - 1 to 2 stars: negative
- **Cleaning:** lowercasing, removing URLs, digits, punctuation and emojis, removing missing values and duplicates, and dropping very short reviews (fewer than 3 words).
- Stop words were deliberately **not removed**, because words like "not" and "never" change the meaning of a review.

### 3. Exploratory data analysis (`notebooks/EDA.ipynb`)
Class distribution, star ratings, review length, top words and bigrams, word clouds, rating trends over time, and the words that most separate positive from negative reviews. Charts are saved in `reports/figures/`.

### 4. Feature engineering
**TF-IDF** with unigrams and bigrams (`sublinear_tf=True`, `min_df=3`, up to 50,000 features), built into the scikit-learn pipeline so the same transformation is applied at prediction time.

### 5. Model training and tuning (`src/train.py`)
- 80/20 stratified train-test split
- Three models compared using 5-fold stratified cross-validation: Logistic Regression, LinearSVC and Complement Naive Bayes
- The best model was tuned with `GridSearchCV` (regularization strength and n-gram range)
- Class weights were used to handle imbalance

### 6. Evaluation
Accuracy, macro F1, a per-class classification report, a confusion matrix, error analysis of misclassified reviews and a plot of the most influential words per class.

### 7. Deployment
The final pipeline is saved with `joblib` and served by a Streamlit app, hosted on Streamlit Community Cloud directly from this GitHub repository.

---

## Results

### Model comparison (5-fold cross-validation)

| Model | CV Accuracy | CV Macro F1 |
|---|---|---|
| Logistic Regression | _fill from reports/model_comparison.csv_ | _fill_ |
| LinearSVC | _fill_ | _fill_ |
| Complement Naive Bayes | _fill_ | _fill_ |

### Final model: Logistic Regression (tuned)

| Metric | Test set |
|---|---|
| Accuracy | **78.22%** |
| Macro F1 | **0.54** |

Per-class results (precision, recall, F1) are in the classification report printed by `src/train.py`. The confusion matrix is saved at `reports/figures/11_confusion_matrix.png`.

### Key observations
- The model performs well on **positive** and **negative** reviews.
- **Neutral** reviews are the hardest class. They are a small, ambiguous class (3-star reviews often mix praise and complaints), which pulls macro F1 well below accuracy.
- Common error types: star rating not matching the text, mixed-sentiment reviews, sarcasm, very short reviews and Hinglish text.

---

## How to run locally

**Prerequisites:** Python 3.11 or newer and Git.

```bash
# 1. Clone the repository
git clone https://github.com/RISHCR7/sentiment-analyzer.git
cd sentiment-analyzer

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/app.py
```

The app opens at `http://localhost:8501`.

### Re-running the full pipeline (optional)

These steps need extra libraries beyond the app's `requirements.txt`:

```bash
pip install google-play-scraper matplotlib seaborn wordcloud ipykernel
python src/collect.py       # scrape reviews
python src/preprocess.py    # clean and label
python src/train.py         # train, tune, evaluate, save model
```

Then open `notebooks/EDA.ipynb` to see the analysis.

---

## Using the app

**Single review tab**
1. Pick an example or type your own review.
2. Click **Analyze**.
3. View the predicted sentiment, confidence and probability chart.

**Bulk CSV analysis tab**
1. Upload a CSV with a column of review text.
2. Select the text column.
3. Click **Run bulk analysis** to see the sentiment split, review length chart, a filterable table and a download button.

---

## Limitations

- Labels come from **star ratings**, which can be noisy (a user may write a complaint and still give 4 stars).
- The **neutral** class is weak because of class imbalance and ambiguity.
- The model does not understand **sarcasm** or **Hinglish / code-mixed** text.
- The training data comes from a single app (Zomato), so performance may drop on other domains.

---

## Future scope

- Fine-tune a transformer model such as DistilBERT and compare it with TF-IDF models
- Support Hinglish and multilingual reviews
- Aspect-based sentiment analysis (for example delivery vs food quality vs app performance)
- Collect reviews from multiple apps and domains
- Real-time review monitoring dashboard
- Manual label verification on a sample of the data

---

## Author

**Rudraksh**
MCA Student
GitHub: [@RISHCR7](https://github.com/RISHCR7)