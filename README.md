# Ridge Regression & SVM Classification Project 

This repository contains my submission for my research project for the **CITS5508 Machine Learning** unit at the University of Western Australia. The research involves an end-to-end machine learning project consisting of both regression and classification tasks, implemented and documented in a single Jupyter Notebook.

## 📁 File

- `Assignment1.ipynb`: Contains all code, outputs, markdown explanations, methodology, and results.

## 🧠 Project Overview

This project is divided into two parts:

### Part 1: Ridge Regression

A comparison of two methods for fitting Ridge Regression models:
- **Closed-form solution**
- **Stochastic Gradient Descent (SGD)** using `sklearn.linear_model.SGDRegressor`

Tasks include:
- Generating synthetic polynomial datasets
- Implementing Ridge Regression from scratch
- Comparing implementations across different polynomial degrees and regularisation strengths (α = 0, 0.1, 100)
- Analyzing performance and scalability

### Part 2: Support Vector Classifier (SVC)

Using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset to:
- Preprocess and explore the data
- Perform hyperparameter tuning using Grid Search with 5-fold cross-validation
- Train and evaluate an `sklearn.svm.SVC` model
- Analyze the model using metrics such as accuracy, precision, and recall
- Discuss real-world applicability of the model

## 🔧 Technologies Used

- Python 3
- NumPy, pandas, matplotlib, seaborn
- scikit-learn
- Jupyter Notebook

## 📊 Key Skills Demonstrated

- Ridge Regression (analytical and SGD-based)
- Polynomial feature engineering
- Machine learning model tuning and evaluation
- Stratified data splitting and cross-validation
- Data visualization and interpretation
- Technical report writing using Markdown and inline code comments

## 📌 Instructions to Run

To ensure the notebook runs correctly:

1. Google Colab.
2. Download the [WDBC dataset](https://doi.org/10.24432/C5DW2B) and place it in the same directory as the notebook.
3. Open `Assignment1.ipynb` and run all cells.

## 📝 License

This project is part of academic coursework and is not intended for commercial use.
