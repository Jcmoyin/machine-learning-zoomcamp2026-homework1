# Credit Risk Prediction Using Machine Learning

## Project Overview

This project uses machine learning to predict the credit risk of applicants based on their demographic, financial, and personal information.

The objective is to identify applicants who may have a higher likelihood of delayed loan or credit repayments. The project uses historical credit records to develop a classification model.

**Note:** This project predicts credit risk as a proxy. It does not make actual credit card approval decisions.

## Dataset

The project uses two datasets:

- `application_record.csv`: Contains applicant demographic, financial, and personal information.
- `credit_record.csv`: Contains applicants' historical credit repayment records.

The datasets were merged using the applicant ID.

The target variable, `credit_risk`, was engineered from the credit repayment status:

- `0`: No recorded repayment status indicating 30 or more days overdue.
- `1`: At least one recorded repayment status indicating 30 or more days overdue.

The prepared dataset contains 36,457 records and 18 columns, including 17 input features and one target variable.

## Project Structure

```text
credit-card-approval-prediction/
│
├── data/
│   └── credit_approval_dataset.csv
│
├── extensive_EDA.ipynb
├── prepare_data.py
├── train.py
├── predict.py
├── pipeline.bin
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset, examine missing values, study the target distribution, and identify relationships between applicant characteristics and credit risk.

The analysis included:

- Dataset inspection and summary statistics.
- Missing-value analysis.
- Target-class distribution.
- Numerical and categorical feature exploration.
- Feature importance analysis.

The target variable is imbalanced, with approximately 88.2% lower-risk records and 11.8% higher-risk records.

## Data Preprocessing

The preprocessing pipeline includes:

- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- Standard scaling for numerical features.
- One-hot encoding for categorical features.
- Stratified train-test splitting.

## Model Development

Three classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Class weighting was used to address the imbalanced target variable.

Random Forest was selected and tuned using RandomizedSearchCV with cross-validation, using F1-score as the optimization metric.

### Best Model Parameters

- `n_estimators`: 100
- `min_samples_leaf`: 2
- `max_features`: 0.5
- `max_depth`: None
- `class_weight`: balanced

## Model Evaluation

The selected Random Forest model achieved the following results on the held-out test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 0.8013 |
| Precision | 0.3236 |
| Recall | 0.6317 |
| F1-score | 0.4280 |
| ROC-AUC | 0.7984 |

The model achieved a recall of approximately 63.17% for the higher-risk class. Its precision of approximately 32.36% indicates that some applicants predicted as higher risk were not actually in the higher-risk class under the project's target definition.

These results highlight the trade-off between identifying higher-risk applicants and minimizing false-positive predictions.

## API Development

A REST API was developed using FastAPI to make predictions with the trained model.

The API provides the following endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks API and model availability |
| `/predict` | POST | Predicts an applicant's credit risk |

The prediction endpoint returns:

- `credit_risk`: Predicted risk class.
- `risk_level`: Higher Risk or Lower Risk.
- `risk_probability`: Predicted probability of belonging to the higher-risk class.

Interactive API documentation is available through FastAPI Swagger UI at `/docs`.

## Docker Deployment

The application was containerized using Docker.

Build the Docker image:

```bash
docker build -t credit-risk-prediction .
```

Run the container:

```bash
docker run -d --name credit-risk-api -p 9696:9696 credit-risk-prediction
```

Test the health endpoint:

```bash
curl http://localhost:9696/health
```

Open the API documentation:

```text
http://localhost:9696/docs
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Docker
- Jupyter Notebook

## How to Run the Project Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project folder:

```bash
cd credit-card-approval-prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn predict:app --reload --port 9696
```

Visit:

```text
http://localhost:9696/docs
```

## Limitations

- The target variable is engineered from historical repayment records and is not an official bank approval label.
- The dataset contains class imbalance.
- The model's precision indicates that false-positive predictions remain a limitation.
- Predictions should be treated as decision-support information rather than final financial decisions.

## Conclusion

This project demonstrates an end-to-end machine learning workflow, covering data preparation, exploratory analysis, model training, evaluation, API development, and Docker containerization.

It provides a foundation for applying machine learning to credit-risk assessment.
