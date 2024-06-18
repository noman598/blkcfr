from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import mord
import json

# Connect to MongoDB
client = MongoClient('mongodb+srv://blackcofferconsulting:Td75gzYeUPToDAPj@cluster0.dv4rmam.mongodb.net/?retryWrites=true&w=majority')
db = client['test']
collection = db['House-price']

# Load data from MongoDB
data = list(collection.find())
df = pd.DataFrame(data)

# Remove the first column
df = df.iloc[:, 1:]
df_numeric = df.select_dtypes(include=[np.number])  # Select only numeric columns
correlation_matrix = df_numeric.corr()
correlation_data = correlation_matrix.to_json(orient='split')
correlation_data = json.loads(correlation_data)  # Parse JSON string into a dictionary

# Define target and preprocess data
target_column = 'smoker'

categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != target_column]
numerical_cols = [col for col in df.columns if col != target_column and col not in categorical_cols]

# Define preprocessing steps
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define ordinal logistic regression model using mord
model = mord.LogisticIT(max_iter=1000)

# Train and evaluate models
results = {}

X = df.drop(target_column, axis=1)
y = df[target_column]

# Check if the target column is not continuous
if df[target_column].dtype == 'object':
    # Convert target column to binary
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
reversed_mapping = {str(value): key for key, value in label_mapping.items()}

bootstrap_auc_scores = []
bootstrap_accuracy_scores = []
bootstrap_log_likelihoods = []
cnt = 1
for bootstrap_iter in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)  # Example split, adjust as needed

    # Preprocess data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    model.fit(X_train_preprocessed, y_train)
    y_pred_proba = model.predict_proba(X_test_preprocessed)

    if len(np.unique(y)) == 2:  # Calculate AUC only if binary classification
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        bootstrap_auc_scores.append(auc)

    y_pred = model.predict(X_test_preprocessed)
    accuracy = accuracy_score(y_test, y_pred)
    bootstrap_accuracy_scores.append(accuracy)
    # Calculate log-likelihood
    log_likelihood = np.sum(np.log(y_pred_proba[np.arange(y_test.shape[0]), y_test]))
    bootstrap_log_likelihoods.append(log_likelihood)
mean_auc = np.mean(bootstrap_auc_scores) if bootstrap_auc_scores else 'N/A'
mean_accuracy = np.mean(bootstrap_accuracy_scores)
mean_log_likelihood = np.mean(bootstrap_log_likelihoods)
results['Ordinal Logistic Regression'] = {
    'Mean AUC': mean_auc,
    'Mean Accuracy': mean_accuracy,
    'Mean Log-Likelihood': mean_log_likelihood,
    'Actual': y_test.tolist(),
    'Predicted': y_pred.tolist(),
    'X_test': X_test.to_dict(orient='records')
}

# Extract coefficients and intercepts
coef_dict = {}
for coef, feat in zip(model.coef_, X.columns):
    coef_dict[feat] = coef
intercept = model.theta_

results['Ordinal Logistic Regression']['Coefficients'] = coef_dict
results['Ordinal Logistic Regression']['Intercept'] = intercept.tolist() if hasattr(intercept, 'tolist') else intercept

# Calculate BIC and AIC
bic = -2 * mean_log_likelihood + np.log(len(df)) * len(X.columns)
aic = -2 * mean_log_likelihood + 2 * len(X.columns)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
results['Ordinal Logistic Regression']['BIC'] = bic
results['Ordinal Logistic Regression']['AIC'] = aic
results['Ordinal Logistic Regression']['fpr'] = fpr.tolist()
results['Ordinal Logistic Regression']['tpr'] = tpr.tolist()
results['Ordinal Logistic Regression']['HeatMap'] = correlation_data

print(json.dumps(results, indent=4))
