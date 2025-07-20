# Smart Disease Predictor System

## Project Overview

The Smart Disease Predictor System is a Streamlit-based web application that leverages various Machine Learning algorithms to predict a potential disease based on user-provided symptoms. This project aims to offer a user-friendly interface for preliminary health assessments, utilizing trained models to provide predictions and insights.

## Live Demo

Experience the app live here: [Click Here](https://jhagauravkr-diseasepredictor.streamlit.app/)


## Features

* **Intuitive User Interface:** Built with Streamlit, providing a clean, modern, and interactive web interface.
* **Multiple ML Algorithms:** Integrates and compares predictions from four popular classification algorithms:
    * Decision Tree
    * Random Forest
    * K-Nearest Neighbors (KNN)
    * Naive Bayes (GaussianNB)
* **Consensus Prediction:** Provides a "Final Outcome" based on a majority vote among the individual model predictions, offering a more robust result.
* **Symptom-based Input:** Users can select up to five symptoms to get a prediction.
* **Interactive Visualizations:** Displays scatter plots to visualize the selected input symptoms and the distribution of symptoms for the predicted disease.
* **Model Performance Metrics:** Shows accuracy and confusion matrices for each algorithm on the test data, providing transparency about model effectiveness.
* **Database Logging:** Stores patient name, selected symptoms, and predicted disease in a local SQLite database for record-keeping.
* **Error Handling:** Includes basic validation for user inputs and handles potential data processing errors gracefully.
* **Efficient Caching:** Utilizes Streamlit's caching mechanisms (`@st.cache_data` and `@st.cache_resource`) to optimize performance by loading data and training models only once.

## How It Works

The system is trained on a dataset of various symptoms and their corresponding diseases. When a user inputs symptoms:
1.  The input is converted into a numerical format consistent with the training data.
2.  Each of the four pre-trained Machine Learning models makes a prediction.
3.  A final prediction is determined by a majority vote (if at least two models agree).
4.  Results, including individual model predictions and the final outcome, are displayed along with relevant plots.
5.  All prediction details are logged into a local SQLite database.

## Setup and Installation

Follow these steps to set up and run the project locally on your machine.

### Prerequisites

* Python 3.7+ (preferably Python 3.8, 3.9, or 3.10)
* `pip` (Python package installer)

