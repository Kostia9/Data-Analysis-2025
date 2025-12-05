# Data-Analysis-2025

This repository contains lab reports and materials for the "Data Analysis" course (2025). The projects cover the full data analysis cycle: from Exploratory Data Analysis (EDA) and Classical Machine Learning to Deep Learning and Generative AI.

## 📂 Repository Structure

### [Lab 1: Exploratory Data Analysis (EDA) & Classification](Lab1/)
**File:** `Lab1.ipynb`
* Data loading, metadata inspection, and handling missing values.
* Visualization: Heatmaps for correlations, histograms, and boxplots relative to the target variable.
* Data normalization.
* Training classifiers: kNN, Decision Tree, Random Forest, AdaBoost, and SVM (using GridSearch to find optimal 'C' and 'gamma').
* Model evaluation: Selecting the best model and displaying classification_report and confusion_matrix.

### [Lab 2: Dimensionality Reduction, Clustering & NLP](Lab2/)
**Files:** `Lab2.ipynb`, `image.jpg`
* Dimensionality Reduction: Applying PCA (comparing performance with Lab 1 models) and t-SNE for 2D visualization.
* Cluster Analysis: Image quantization using K-Means (reducing colors to 64, 32, 16, and 8 levels).
* Text Processing (NLP): Preprocessing (stop-words/punctuation removal), WordCloud visualization, vectorization (e.g., Tfidf), and text classification.

### [Lab 3: Deep Learning](Lab3/)
**Files:** `Lab3_1.ipynb`, `Lab3_2.ipynb`, `Lab3_3.ipynb`
* Introduction to Neural Networks (PyTorch/TensorFlow).
* **Part 1:** Building and training a Multi-Layer Perceptron (MLP).
* **Part 2:** Convolutional Neural Networks (CNN) for image classification.
* **Part 3:** Recurrent Neural Networks (RNN/GRU/LSTM) for text classification.

### [Lab 4: Modern Architectures & Generative AI](Lab4/)
**Files:** `Lab4_1.ipynb`, `Lab4_2.ipynb`, `Lab4_3.ipynb`
* **Part 1:** Working with Transformers (GPT-like architecture) for text generation.
* **Part 2:** Diffusion Models for image generation.
* **Part 3:** Using pre-trained models (Hugging Face).
