# AT&T spam detector

This project aims to develop an **automated spam detector for SMS messages** to protect users from unwanted messages. 

We build two text classification models using deep learning techniques:
- A **simple deep learning model**.
- A **transfer learning model**.

## Dataset

The dataset contains 5,572 SMS messages written in English, each labeled as either 'ham' or 'spam'.

[Link to the dataset](https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Deep+Learning/project/spam.csv)

## Project workflow

- **Exploratory data analysis**  

- **Data preprocessing**
    - Text cleaning

    - Label encoding: babels ("ham" and "spam") are encoded into integers

    - Data splitting into training and test sets

    - Text tokenization and token encoding


- **Model training**
    - Build a simple deep learning model
    - Use a pre-trained model in transfer learning to improve performance (DistilBERT).

- **Model evaluation and comparison**  
Evaluate the models on a test set and compare performance (precision, recall, F1-score, accuracy)

## Deliverables

- Jupyter notebook: `spam_detector.ipynb`, created using Google Colab
- Visualisations stored in the `exports/` folder

## Usage
If you are using Google Colab, you will need to mount your Google Drive to save results (charts or trained models):
````
from google.colab import drive
drive.mount('/content/drive')
````


## Tech stack
| **Category**       | **Technology / Library**                          |
|---------------------|------------------------------------------|
| Programming language           | Python                       |
| Data processing                 | pandas, numpy                         |
| Text processing                 | spaCy, re                        |
| Deep learning                | pytorch, transformers (DistilBERT for transfer learning)                        |
| Data visualisation    | matplotlib, plotly                     |


## Key insights

This dataset is highly imbalanced, with a much smaller proportion of spam messages (13%) compared to ham messages (87%). 

| Model            | Precision  (test)| Recall (test)| F1  (test)    | Accuracy  (test)|
|------------------|-----------|---------|--------|----------|
| Baseline model   | 0.983    | 0.869  | 0.922 | 0.982   |
| DistilBERT       | 0.983    | 0.908  | 0.944 | 0.986   |


 **The DistilBERT model outperforms the baseline model**: it maintains excellent precision while reducing false negatives (improving the recall and F1-score).
 
 **The drawback of using transfer learning is its execution time**: on Google Colab, while the baseline model training takes only around 5 seconds, DistilBERT may require 3 minutes to complete the same task.**