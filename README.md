# SEG4300 Applied Machine Learning - Assignments Portfolio

- **Name:** Mohamed-Obay Alshaer
- **Student Number:** 300170489
- **Course:** SEG4300 - Applied Machine Learning
- **Term:** Winter 2025

## Overview

This repository contains my solutions for Assignments 2-5 for SEG4300 (Applied Machine Learning). Each assignment explores different aspects of machine learning, from exploratory data analysis to deep learning, using datasets from the Hugging Face Datasets Library.

## Assignments

### Assignment 2
- **Dataset:** IMDb Reviews Dataset (stanfordnlp/imdb)
- **Objective:** Perform comprehensive exploratory data analysis on a text dataset
- **Techniques:** Statistical analysis, data visualization, text analysis, data quality assessment
- **Key Components:**
  - Text preprocessing and feature extraction
  - Word frequency analysis and word clouds
  - Sentiment distribution analysis
  - Data quality validation using Great Expectations

### Assignment 3
- **Dataset:** Wisconsin Breast Cancer Dataset (scikit-learn/breast-cancer-wisconsin)
- **Objective:** Train and evaluate a machine learning model using Scikit-learn
- **Techniques:** Data preprocessing, feature scaling, logistic regression, model evaluation
- **Key Components:**
  - Binary classification of malignant vs. benign tumors
  - Feature importance analysis
  - Performance evaluation with multiple metrics
  - Confusion matrix and ROC curve analysis

### Assignment 4
- **Dataset:** AG News Dataset (fancyzhx/ag_news)
- **Objective:** Apply clustering techniques and train a supervised model to predict clusters
- **Techniques:** Text vectorization (TF-IDF), dimensionality reduction, K-means clustering
- **Key Components:**
  - Elbow method for optimal cluster determination
  - Visualization of clusters using t-SNE and PCA
  - Interpretation of discovered topic clusters
  - Supervised classification of cluster assignments

### Assignment 5
- **Dataset:** CIFAR-10 Image Dataset (uoft-cs/cifar10)
- **Objective:** Build, train, and evaluate a deep learning model for image classification
- **Techniques:** Convolutional Neural Networks (CNN), data augmentation, transfer learning
- **Key Components:**
  - Multi-class image classification
  - Model architecture with convolutional blocks
  - Training with early stopping and learning rate scheduling
  - Detailed performance analysis by class

## Dependencies

Each notebook contains the necessary installation commands, but the main dependencies include:
- Python 3.8+
- Hugging Face Datasets
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- PyTorch and Torchvision
- NLTK (for text processing)
- tqdm (for progress tracking)

## Running the Notebooks

Each assignment is contained in its own Jupyter notebook. To run:

1. Clone this repository
2. Install the required dependencies (pip install requirements.txt)
3. Launch Jupyter Lab or Notebook
4. Open the desired assignment notebook
5. Execute the cells in sequence