# Day 2 – OOP Machine Learning Pipeline (Heart Disease Prediction)

A modular, scalable, production-grade ML architecture built using Python OOP

*🚀 Project Overview*

Day 2 focuses on building a fully modular, object-oriented machine learning system — the same engineering style used at Google, Amazon, Meta, Netflix, and Apple.

We refactor a classical ML workflow into clean, testable, reusable components:

Dataset Loader

Tabular Preprocessor

Model Wrapper (Logistic Regression / Random Forest)

Training Orchestrator

Configuration System (dataclasses)

Logging for debugging and production readiness

This creates an industry-grade ML pipeline that can easily scale to new datasets, models, and deployment systems.

🧠 Problem Statement

We use the Heart Disease Cleveland Dataset to predict whether a patient is likely to have heart disease.

Input: 13 medical features (age, sex, cp, chol, trestbps, thalach, etc.)

Output: condition → 1 (disease), 0 (no disease)

Task Type: Binary Classification

Algorithms: Logistic Regression / Random Forest

🏗️ Project Architecture (FAANG-style)

heart_disease_oop_ml/
│
├── data/

│   └── heart_cleveland_upload.csv

│

├── src/

│   ├── data/

│   │   └── dataset_loader.py

│   │

│   ├── preprocessing/

│   │   └── preprocessor.py

│   │

│   ├── models/

│   │   └── model_wrapper.py

│   │

│   ├── training/

│   │   └── trainer.py

│   │

│   └── utils/

│       ├── config.py


│       └── logging_utils.py

│

├── main.py

├── README.md

└── requirements.txt

This structure separates responsibilities clearly — a key requirement for scalable ML systems.

**🧩 Main Components**
🔹 1. DataConfig, ModelConfig, TrainingConfig

Defined using Python dataclasses, enabling:

Reproducibility

Clear configuration

No hardcoded values

Easy switching of models / parameters

🔹 2. DatasetLoader

Handles:

Reading CSV

Splitting into train/test

Extracting target column

Returning numpy arrays

🔹 3. TabularPreprocessor

StandardScaler applied with fit only on training data → prevents data leakage.

🔹 4. ModelWrapper

Wraps ML models behind a unified interface:

.fit()

.predict()

.predict_proba()

.evaluate()

Supports:

Logistic Regression

Random Forest

Adding new models becomes trivial.

🔹 5. Trainer

Orchestrates the entire ML lifecycle:

Load data

Preprocess

Train model

Evaluate

Log results

📊 Model Performance

Using Logistic Regression:

Metric	Score
Accuracy	0.9166
ROC-AUC	0.9531

A high ROC-AUC indicates excellent medical prediction performance.

▶️ How to Run
python main.py


All configuration is handled in:

src/utils/config.py

🧪 Next Extensions (Recommended)

SHAP explainability

Feature importance visualization

Add XGBoost / LightGBM models

Build API using FastAPI

Unit tests for the pipeline

Hyperparameter tuner module

🏅 Skills Demonstrated

Professional ML Engineering

Object-Oriented Design

Clean Architecture

Config-Driven Development

Logging & Reproducibility

Production-ready pipelines

📦 Dependencies

Install required packages:

pip install -r requirements.txt

⭐ Summary

Day 2 delivers a real machine learning system, not just a notebook.
You built something that mirrors FAANG-level ML engineering practices:

Modular

Scalable

Testable

Reusable

Clean OOP design

This is exactly the kind of structure companies want.
