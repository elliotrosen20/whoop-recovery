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