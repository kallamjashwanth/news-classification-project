# News Classification Project

## Overview
This project builds a machine learning pipeline to classify news articles into categories using the AG News dataset from Kaggle.

The model uses:
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression classifier

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
news_classification_project/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── train.py
│ ├── evaluate.py
│ ├── config.py
│
├── models/
├── results/
├── requirements.txt
├── README.md
└── main.py

## Installation
1. Create virtual environment (optional):
python -m venv venv
venv\Scripts\activate (Windows)


2. Install dependencies:
pip install -r requirements.txt

## How to Run
Run the project using:
python main.py

## Results

Model Used: Logistic Regression  
Feature Extraction: TF-IDF  

Accuracy Achieved: ~90.4%

Classification report is saved in:
results/metrics.txt


