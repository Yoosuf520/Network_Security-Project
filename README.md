# Network Security Project — Phishing Website Detection

An end-to-end machine learning pipeline that detects **phishing websites** from URL/webpage-based features. Built as a modular MLOps project — not a single notebook — with separate stages for data ingestion, validation, transformation, model training, experiment tracking, and API-based serving.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Folder Structure](#folder-structure)
- [Pipeline Stages](#pipeline-stages)
- [Model Training](#model-training)
- [Experiment Tracking (MLflow + DagsHub)](#experiment-tracking-mlflow--dagshub)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Running the Project](#running-the-project)
- [API Endpoints](#api-endpoints)
- [Future Improvements](#future-improvements)

---

## Problem Statement

Phishing websites imitate legitimate ones to steal user credentials and sensitive data. This project frames phishing detection as a **binary classification problem**: given a set of features extracted from a website, predict whether it is:

- `1` → Phishing
- `0` → Legitimate

## Dataset

- Source: UCI-style Phishing Websites dataset (`Network_data/phisingData.csv`)
- 30 pre-engineered input features + 1 target column (`Result`)
- Features capture URL structure, domain properties, and page behavior, e.g.:
  - `having_IP_Address`, `URL_Length`, `Shortining_Service`, `having_At_Symbol`
  - `SSLfinal_State`, `Domain_registeration_length`, `age_of_domain`
  - `Request_URL`, `URL_of_Anchor`, `Links_in_tags`, `Abnormal_URL`
  - `web_traffic`, `Page_Rank`, `Google_Index`, `Statistical_report`
- Schema is enforced via `data_schema/schema.yaml`, which lists every expected column and its type.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost) |
| Data storage | MongoDB |
| Experiment tracking | MLflow + DagsHub |
| API | FastAPI, Uvicorn |
| Data validation | Custom schema checks + KS-test drift detection (SciPy) |
| Config/Logging | Custom exception handling & logging module |

## Project Architecture

```
MongoDB  →  Data Ingestion  →  Data Validation  →  Data Transformation  →  Model Training  →  final_model/ (model.pkl + preprocessor.pkl)
                                                                                                      │
                                                                                                      ▼
                                                                                          FastAPI (/train, /predict)
```

Each stage takes a **config** object (settings) and produces an **artifact** object (its output), which becomes the input to the next stage. This keeps every stage independently testable and swappable.

## Folder Structure

```
Network_Security-Project/
├── Network_data/                     # Raw dataset (phisingData.csv)
├── data_schema/
│   └── schema.yaml                   # Expected columns & dtypes
├── final_model/
│   ├── model.pkl                     # Best trained model
│   └── preprocessor.pkl              # Fitted KNN imputer pipeline
├── logs/                             # Timestamped run logs
├── networksecurity/
│   ├── components/
│   │   ├── data_ingestion.py         # MongoDB → CSV → train/test split
│   │   ├── data_validation.py        # Schema checks + data drift (KS-test)
│   │   ├── data_transformation.py    # KNN imputation
│   │   └── model_training.py         # Train, tune, select, track best model
│   ├── entity/
│   │   ├── config_entity.py          # Config classes per stage
│   │   └── artifact_entity.py        # Output/artifact classes per stage
│   ├── constant/training_pipeline/   # All constants (paths, params, thresholds)
│   ├── exception/exception.py        # Custom exception class
│   ├── logging/logger.py             # Logging setup
│   ├── pipeline/
│   │   └── training_pipeline.py      # Orchestrates all stages end-to-end
│   └── utils/
│       ├── main_utils/utils.py       # Save/load objects, GridSearchCV evaluation
│       └── ml_utils/
│           ├── metrics/classification_metrics.py  # F1, precision, recall
│           └── model/estimator.py    # NetworkModel wrapper (preprocess + predict)
├── prediction_output/                # Saved prediction results
├── templates/table.html              # HTML table for prediction results
├── app.py                            # FastAPI app (train & predict routes)
├── main.py                           # Script to run pipeline stages directly
├── push_data.py                      # Uploads local CSV into MongoDB
├── requirements.txt
└── setup.py
```

## Pipeline Stages

### 1. Data Ingestion
- Reads data from a MongoDB collection into a DataFrame
- Saves it to a feature store CSV
- Splits into train/test sets (80/20)

### 2. Data Validation
- Confirms the number of columns matches `schema.yaml`
- Confirms numerical columns exist
- Runs a **Kolmogorov–Smirnov test** between train and test distributions per column to flag data drift, writing a `report.yaml`

### 3. Data Transformation
- Handles missing values using a **KNN Imputer** (`n_neighbors=3`)
- Fits the imputer on training data, transforms both train and test sets
- Saves the fitted preprocessor object for reuse at inference time

### 4. Model Training
- Trains and tunes 5 candidate models with `GridSearchCV` (3-fold CV):
  - Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost
- Selects the best model based on **F1-score** on the test set
- Evaluates the final model with **F1, Precision, and Recall**
- Logs parameters/metrics/model to **MLflow**, backed by **DagsHub**
- Saves the winning model and preprocessor to `final_model/`

## Model Training

| Model | Tuned hyperparameters |
|---|---|
| Decision Tree | `criterion` |
| Random Forest | `n_estimators` |
| Gradient Boosting | `learning_rate`, `subsample`, `n_estimators` |
| Logistic Regression | default |
| AdaBoost | `learning_rate`, `n_estimators` |

Model selection is based on **F1-score** (not a regression metric), which is appropriate here since phishing detection benefits from balancing precision and recall rather than optimizing raw accuracy alone.

## Experiment Tracking (MLflow + DagsHub)

Every training run logs:
- `f1_score`, `precision`, `recall`
- The trained model artifact

Tracking is routed through DagsHub via:

```python
import dagshub
dagshub.init(repo_owner='Yoosuf520', repo_name='Network_Security-Project', mlflow=True)
```

To sanity-check the connection independently of the full pipeline, run:

```bash
python dagshub_mlflow_test.py
```

Then check the **Experiments** tab on the DagsHub repo for a run named `connection_test`.

## Setup & Installation

```bash
git clone https://github.com/Yoosuf520/Network_Security-Project.git
cd Network_Security-Project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```
MONGO_DB_URL=<your MongoDB connection string>
MONGODB_URL_KEY=<your MongoDB connection string>
```

> Note: `push_data.py` / `data_ingestion.py` use `MONGO_DB_URL`, while `app.py` uses `MONGODB_URL_KEY`. Set both to the same connection string.

## Running the Project

**1. Push the dataset into MongoDB (first time only):**
```bash
python push_data.py
```

**2. Run the full training pipeline directly:**
```bash
python main.py
```

**3. Or start the API and trigger training via endpoint:**
```bash
python app.py
```
Then visit `http://localhost:8000/docs` for interactive API docs.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Redirects to `/docs` |
| GET | `/train` | Runs the full training pipeline |
| POST | `/predict` | Upload a CSV of feature rows, get phishing/legitimate predictions rendered as an HTML table |

## Future Improvements

- Broader classification metrics during model selection (e.g. ROC-AUC) alongside F1
- Handle class imbalance explicitly (e.g. class weights or SMOTE)
- Add unit tests for each pipeline component
- Add model versioning / a model registry
- Add authentication to the `/train` and `/predict` endpoints
- Re-introduce Docker + cloud deployment (S3 sync) once the core pipeline is stable

---

*This project follows a modular MLOps pipeline pattern — each stage is independently configurable, testable, and produces a typed artifact consumed by the next stage.*