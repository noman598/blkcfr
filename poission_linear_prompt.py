




def poisson_linear_pmpt(schema_det, total_row, total_col, target, new_table, goal_name, threshold, models_names):

    prompt = f'''

    Objective:
I need assistance in training a machine learning model utilizing a dataset stored within MongoDB. My goal is to {goal_name}, while ensuring that certain columns with string values undergo preprocessing before model training. Additionally, after training, I aim to retrieve the coefficients of the trained model with respect to columns.
Instructions:
    Please provide the necessary code snippets focusing solely on the code.
    Connect to MongoDB using Python with the following connection details:

        use the below code to generate code for some condition - 
        take collection as {new_table} and target as {target}. 
        
        check threshold for this value - {threshold}. 
        here is more detail about the dataset - 
        schemas - {schema_det}, total_row - {total_row}, total_col - {total_col}


from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import PoissonRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import json
import warnings
import sys
import os
import statsmodels.api as sm

# Connect to MongoDB
client = MongoClient('mongodb+srv://blackcofferconsulting:Td75gzYeUPToDAPj@cluster0.dv4rmam.mongodb.net/?retryWrites=true&w=majority')
db = client['test']
collection = db['{new_table}']

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
target_column = '{target}'


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

# Define Poisson and Multiple regression models
poisson_model = PoissonRegressor(max_iter=1000)
multiple_regression_model = LinearRegression()

# Train and evaluate models
results = {{}}

X = df.drop(target_column, axis=1)
y = df[target_column]

# Check if the target column is not continuous
if df[target_column].dtype == 'object':
    # Convert target column to binary
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
import warnings
import sys
import os

# Function to train and evaluate a model
def evaluate_model(model, model_name):
    bootstrap_auc_scores = []
    bootstrap_accuracy_scores = []
    bootstrap_log_likelihoods = []
    cnt = 1
    for bootstrap_iter in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Preprocess data
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        pipeline = Pipeline(steps=[('model', model)])
        pipeline.fit(X_train_preprocessed, y_train)
        

        # Calculate AUC

        y_pred = pipeline.predict(X_test_preprocessed)
        y_pred_binary = list(map(lambda x: 1 if x > threshold else 0, y_pred))
       

        y_test_binary = list(map(lambda x: 1 if x > threshold else 0, y_test))
        auc = roc_auc_score(y_test_binary, y_pred_binary)
        bootstrap_auc_scores.append(auc)

        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        bootstrap_accuracy_scores.append(accuracy)

        # Redirect stdout to suppress optimization messages
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            # Calculate log-likelihood
            poisson_model_sm = sm.Poisson(y_test, sm.add_constant(y_pred)).fit()
            log_likelihood = poisson_model_sm.llf
            bootstrap_log_likelihoods.append(log_likelihood)
        finally:
            # Reset stdout
            sys.stdout.close()
            sys.stdout = original_stdout

    mean_auc = np.mean(bootstrap_auc_scores)
    mean_accuracy = np.mean(bootstrap_accuracy_scores)
    mean_log_likelihood = np.mean(bootstrap_log_likelihoods)

    # Coefficients
    coef_dict = {{feat: coef for feat, coef in zip(X.columns, pipeline.named_steps['model'].coef_)}}
    intercept = pipeline.named_steps['model'].intercept_

        
    # Fit the statsmodels logistic regression for p-values
    X_train_preprocessed_sm = sm.add_constant(X_train_preprocessed)
    model_sm = sm.MNLogit(y_train, X_train_preprocessed_sm).fit()

    # Get p-values
    p_values = model_sm.pvalues

    # Create p-values dictionary
    p_values_dict = {{}}
    for p_value, feat in zip(p_values.tolist(), X.columns.tolist()):
        p_values_dict[feat] = p_value


    # Calculate BIC and AIC
    n = len(df)
    p = len(X.columns)
    aic = -2 * mean_log_likelihood + 2 * p
    bic = -2 * mean_log_likelihood + np.log(n) * p
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)

    results[model_name] = {{
        'Mean AUC': mean_auc,
        'Mean Accuracy': mean_accuracy,
        'Mean Log-Likelihood': mean_log_likelihood,
        'Actual': y_test.tolist(),
        'Predicted': y_pred.tolist(),
        'X_test': X_test.to_dict(orient='records')
        'Coefficients': coef_dict,
        'Intercept': intercept,
        'BIC': bic,
        'AIC': aic,
        'tpr':tpr.tolist(),
        'fpr':fpr.tolist(),
        'HeatMap':correlation_data,
        'P-values':p_values_dict
       

    }}
# Evaluate Poisson regression model
evaluate_model(poisson_model, 'Poisson Regression')

# Evaluate Multiple regression model
evaluate_model(multiple_regression_model, 'Multiple Regression')

# Print JSON output
result = json.dumps(results, indent=4)
print(result)

''' 
    return prompt