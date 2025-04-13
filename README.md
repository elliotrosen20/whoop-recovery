---
title: Whoop Recovery
emoji: 🏆
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Predicts recovery score based on biometric data from WHOOP
---

## Whoop Recovery Prediction

This tool predicts a user's WHOOP recovery score using biometric data such as heart rate, respiratory rate, and sleep metrics. It’s based on an XGBoost model trained on real data and serves as a demonstration of how biometric signals can be used to infer recovery readiness.

## Live Demo

You can interact with the model via the [Hugging Face Space](https://huggingface.co/spaces/elliotrosen/whoop-recovery) by adjusting sliders for key input features and instantly seeing the predicted score.

## Neural Net Model

For this project, I also created a Neural Network (NN) model that predicts the WHOOP recovery score using biometric data. Here’s an overview of how the model is built:

### 1. Preprocessing
- Data is cleaned by removing missing values and certain unnecessary columns.
- Specific columns related to sleep stages (Deep, REM, and Light sleep) are used to calculate sleep ratios.
- Numerical features are filled with the median value to handle missing data.
- Irrelevant columns such as Cycle start time, Cycle end time, and other non-informative fields are dropped.

### 2. Target Variable
- The target variable is the **recovery score** (Recovery score %), which is predicted based on the cleaned features.

### 3. Tabular Data Setup
- The dataset is split into training and validation sets using `RandomSplitter`.
- `TabularPandas` from `fastai` is used to handle both categorical and continuous variables.
- Data transformations such as categorifying, filling missing values, and normalizing are applied to prepare the dataset for training.

### 4. Model Architecture
- A neural network model is built using `fastai`’s `tabular_learner`. The model architecture consists of two fully connected layers with 10 nodes each.
- The `EarlyStoppingCallback` is used to prevent overfitting by stopping training when the validation loss stops improving.

### 5. Training
- A learning rate finder is used to determine an optimal learning rate for training.
- The model is trained for 35 epochs with a learning rate of **0.02** using the `fit_one_cycle` method.

### 6. Evaluation
- After training, the model’s performance is evaluated on the test set using various metrics:
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
  - **R²** (Coefficient of Determination)
- The trained model is saved as a `.pkl` file for future use.

### 7. Results
- The neural network model achieved strong test set performance, with **RMSE**, **MAE**, and **R²** metrics indicating high accuracy.
#### Test Set Performance
- **RMSE**: 6.6193  
- **MAE**: 5.3738  
- **R²**: 0.9097

The neural network model, like the XGBoost model, can predict the WHOOP recovery score, but it uses a different architecture, offering a good comparison of model performance.

## XGBoost Model



## Project Details

- Built with: Python, XGBoost, Gradio
- Training approach: 20% test split on a single user’s WHOOP dataset
- Evaluation: K-fold cross-validation on the training set

### Metrics on the Test Set:

- RMSE: 6.62
- MAE: 5.37
- R²: 0.91

## Notebooks

Notebooks demonstrating the training process and model experimentation are available on [Kaggle](#) (add link here). They include data preprocessing, feature engineering, and model evaluation.

## Local Setup

Clone the repo and run:

```bash
pip install -r requirements.txt
python app.py