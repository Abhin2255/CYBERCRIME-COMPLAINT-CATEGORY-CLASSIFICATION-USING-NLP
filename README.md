# Cybercrime Complaint Category Classification using NLP

A machine learning pipeline that automatically classifies cybercrime complaints (free-text descriptions) into crime categories such as fraud, phishing, account hacking, and fake profiles. It combines TF-IDF text features with engineered features and compares three classifiers, then tunes and saves the best model for prediction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Pipeline](#project-pipeline)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Evaluation](#models-and-evaluation)
- [Results](#results)
- [Saved Artifacts](#saved-artifacts)
- [Making Predictions](#making-predictions)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

Cybercrime portals receive a large volume of complaints written in everyday language. Manually sorting them into categories is slow and inconsistent. This project trains a text classifier that reads a complaint and predicts its category, which can help with faster triage and routing.

## Features

- Automatic detection of the complaint text column and the category column in the CSV
- Data cleaning: duplicate removal, missing value handling, type conversion
- NLP preprocessing: lowercasing, URL/email/number/punctuation removal, stopword removal, lemmatization
- Engineered features: text length, word count, URL count, number count, special character count, and a cybercrime keyword count (e.g. OTP, bank, phishing, blackmail)
- TF-IDF with unigrams and bigrams (up to 10,000 features), stacked with scaled engineered features
- Comparison of Logistic Regression, Linear SVM, and Random Forest
- Visualizations: category distribution, word count distribution, model comparison chart, confusion matrix
- 5-fold stratified cross-validation and grid search for hyperparameter tuning
- Model persistence with `joblib` and a ready-to-use `predict_category()` function

## Project Pipeline

```
Raw CSV
  -> Column detection and cleaning
  -> EDA (category and length distributions)
  -> NLP preprocessing (clean, tokenize, remove stopwords, lemmatize)
  -> Feature engineering (TF-IDF + numeric features)
  -> Train/test split (80/20, stratified)
  -> Train and compare models
  -> Cross-validation and GridSearchCV (LinearSVC)
  -> Final evaluation and model saving
  -> Prediction on new complaints
```

## Tech Stack

- Python 3.9+
- pandas, NumPy, SciPy
- scikit-learn
- NLTK
- Matplotlib, Seaborn
- joblib
- XGBoost (imported; not used in the current training flow)

## Dataset

The script expects a CSV file of cybercrime complaints containing:

| Column type | Accepted column names (auto-detected) |
|---|---|
| Complaint text | `complaint_text`, `complaint`, `text`, `description`, `complaint_description`, `message`, `content` |
| Category label | `category`, `label`, `class`, `target`, `crime_category`, `crime_type`, `complaint_category`, `type` |

Column names are normalized (trimmed, lowercased, spaces replaced with underscores) before detection. If no match is found, the script prints the available columns and asks you to set `TEXT_COLUMN` / `TARGET_COLUMN` manually.

> The dataset is not included in this repository. Place your CSV in the project root and update the `FILE` variable in the script.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. (Recommended) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install pandas numpy scipy scikit-learn nltk matplotlib seaborn joblib xgboost
   ```

   NLTK data (`stopwords`, `wordnet`, `omw-1.4`) is downloaded automatically on first run.

## Usage

1. Put your dataset in the project folder.
2. Set the dataset path at the top of the script:

   ```python
   FILE = "your_dataset.csv"
   ```

3. Run the training script:

   ```bash
   python <your_script_name>.py
   ```

The script prints dataset details, trains and compares the models, shows plots, saves the final model, and runs a sample prediction.

## Models and Evaluation

| Model | Key settings |
|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight="balanced"` |
| Linear SVM (LinearSVC) | `C=1.0`, `class_weight="balanced"` |
| Random Forest | `n_estimators=200`, `class_weight="balanced"` |

Each model is evaluated on a held-out 20% test set using accuracy, weighted precision, weighted recall, and weighted F1. The Linear SVM is then cross-validated (5-fold stratified) and tuned with `GridSearchCV` over `C in {0.1, 0.5, 1, 2, 5}` using weighted F1. The best estimator is used as the final model.

## Results

Fill in after running the script on your dataset:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | - | - | - | - |
| Linear SVM | - | - | - | - |
| Random Forest | - | - | - | - |
| Tuned Linear SVM (final) | - | - | - | - |

Best parameters: `C = -`  |  Mean CV F1: `-`

You can add the generated plots (category distribution, model comparison, confusion matrix) to a `images/` folder and embed them here:

```markdown
![Model Comparison](images/model_comparison.png)
![Confusion Matrix](images/confusion_matrix.png)
```

## Saved Artifacts

After training, three files are written to the project folder:

| File | Purpose |
|---|---|
| `cybercrime_classifier.pkl` | Final tuned Linear SVM model |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `feature_scaler.pkl` | Fitted scaler for engineered features |

## Making Predictions

Inside the script, use `predict_category()`:

```python
complaint = (
    "Someone created a fake Instagram account "
    "using my photos and is messaging my friends."
)

print(predict_category(complaint))
```

To use the saved model in another script, load all three artifacts and repeat the same steps: clean the text, compute TF-IDF, compute and scale the six engineered features, stack them, then call `model.predict`.

```python
import joblib

model = joblib.load("cybercrime_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("feature_scaler.pkl")
```

## Project Structure

```
.
├── <your_script_name>.py        # Training, evaluation, and prediction pipeline
├── your_dataset.csv             # Dataset (not included)
├── cybercrime_classifier.pkl    # Generated after training
├── tfidf_vectorizer.pkl         # Generated after training
├── feature_scaler.pkl           # Generated after training
└── README.md
```

## Limitations

- Performance depends on the quality and label consistency of the dataset.
- The keyword feature uses a fixed English keyword list and will not generalize to other languages or slang.
- Bag-of-words TF-IDF does not capture deeper context or word order.
- Rare categories may be predicted less reliably, even with balanced class weights.

## Future Improvements

- Try transformer models such as BERT or DistilBERT for better context understanding
- Add multilingual support for regional-language complaints
- Use `Pipeline` objects so preprocessing and model are saved as a single artifact
- Serve the model through a web interface (e.g. Gradio or Streamlit) or a REST API
- Add explainability (e.g. top TF-IDF terms per category, SHAP)


