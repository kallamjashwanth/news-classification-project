# News Classification Project

## Overview
This project builds a machine learning pipeline to classify news articles into categories using the AG News dataset from Kaggle.

The model uses:
- Text preprocessing
- TF-IDF feature extraction with unigrams and bigrams
- Linear Support Vector Machine (SVM) classifier

## Dataset
Dataset used: AG News (Kaggle)

It contains 4 categories:
1. World
2. Sports
3. Business
4. Sci/Tech

Files used:
- train.csv
- test.csv

## Project Structure
```bash
news-classification-project/
│
├── data/
│   ├── raw/                
│   └── processed/          
│
├── src/
│   ├── data_preprocessing.py   
│   ├── feature_engineering.py  
│   ├── train.py                
│   ├── evaluate.py             
│   └── config.py               
│
├── models/
│   ├── news_classifier.pkl     
│   └── tfidf_vectorizer.pkl    
│
├── results/
│   └── metrics.txt             
│
├── requirements.txt
├── README.md
└── main.py                  
```

## Installation
1. Create virtual environment (optional):
```bash
python -m venv venv
venv\Scripts\activate (Windows)
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
Run the project using:
```bash
python main.py
```

## Results

Model Used: Logistic Regression  
Feature Extraction: TF-IDF  

Accuracy Achieved: 90.578

Classification report is saved in:
results/metrics.txt


