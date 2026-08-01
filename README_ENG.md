# 🩺 Diabetes Prediction with MLflow & CI/CD

## Project Overview

This project demonstrates an end-to-end machine learning workflow for diabetes prediction using **MLflow**, **GitHub Actions**, and **Docker**.

The workflow automates model training, experiment tracking, artifact logging, Docker image creation, and container deployment, providing a reproducible and production-ready machine learning pipeline.

---

## Project Objectives

The objectives of this project are:

- Build a machine learning model for diabetes prediction.
- Track machine learning experiments using MLflow.
- Automate training through MLflow Projects.
- Implement Continuous Integration (CI) using GitHub Actions.
- Build a Docker image automatically from the trained model.
- Publish the Docker image to Docker Hub.

---

## Dataset

The project uses a preprocessed diabetes dataset stored in:

```
diabetes_preprocessing.csv
```

The dataset contains numerical features used to predict whether a patient has diabetes.

---

## Machine Learning Pipeline

```
Preprocessed Dataset
          │
          ▼
     Train/Test Split
          │
          ▼
 Logistic Regression
          │
          ▼
 Model Evaluation
(Accuracy & Confusion Matrix)
          │
          ▼
 MLflow Tracking
          │
          ▼
 MLflow Model
          │
          ▼
 GitHub Actions CI
          │
          ▼
 Docker Image Build
          │
          ▼
 Docker Hub
```

---

## Technologies Used

### Machine Learning

- Python
- Scikit-learn
- Logistic Regression
- Pandas
- NumPy

### Experiment Tracking

- MLflow

### Visualization

- Matplotlib
- Seaborn

### DevOps / MLOps

- GitHub Actions
- Docker
- Docker Hub
- Conda

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── ci-mlflow.yml
├── MLProject
├── modelling.py
├── conda.yaml
├── diabetes_preprocessing.csv
├── Dockerhub.txt
└── README.md
```

---

## ML Pipeline

The training workflow performs the following steps:

1. Load the preprocessed diabetes dataset.
2. Split the dataset into training and testing sets.
3. Train a Logistic Regression model.
4. Evaluate the model using Accuracy Score.
5. Generate a Confusion Matrix.
6. Log experiment metadata with MLflow.
7. Save the trained model as an MLflow Model.

---

## Experiment Tracking

MLflow automatically records:

- Parameters
- Metrics
- Trained Model
- Confusion Matrix
- Artifacts
- Execution Run

The project uses:

```
mlflow.autolog()
```

to simplify experiment tracking.

---

## Continuous Integration

A GitHub Actions workflow automatically performs:

- Repository checkout
- Python environment setup
- Dependency installation
- MLflow Project execution
- Artifact upload
- Docker image creation
- Docker Hub deployment

The workflow runs automatically on every push to the **main** branch.

---

## Docker Deployment

After the training process completes successfully, the workflow automatically:

- Builds a Docker image from the MLflow model.
- Tags the Docker image.
- Pushes the image to Docker Hub.

This enables the trained model to be deployed consistently across environments.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/username/diabetes-mlflow-project.git
```

Navigate to the project directory:

```bash
cd diabetes-mlflow-project
```

Install dependencies:

```bash
pip install mlflow
```

Run the MLflow Project:

```bash
mlflow run .
```

---

## Skills Demonstrated

- Machine Learning
- Logistic Regression
- Model Evaluation
- MLflow Tracking
- MLflow Projects
- Experiment Management
- GitHub Actions
- Continuous Integration (CI)
- Docker
- Docker Hub
- MLOps Fundamentals

---

## Future Improvements

Potential improvements include:

- Hyperparameter tuning.
- Multiple machine learning models.
- Model comparison experiments.
- Continuous Deployment (CD).
- REST API deployment.
- Model monitoring.

---

## Author

**Aprilia Melani**

Machine Learning Engineer | Data Analyst | AI Enthusiast
