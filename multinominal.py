

def multinominal_pmpt(schema_det, total_row, total_col, target, new_table, goal_name, threshold, models_names):

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
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


# Connect to MongoDB
client = MongoClient('mongodb+srv://blackcofferconsulting:Td75gzYeUPToDAPj@cluster0.dv4rmam.mongodb.net/?retryWrites=true&w=majority')
db = client['test']
collection = db[{new_table}]

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
target_column = {target}


print("Hello")
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

# Define multinomial logistic regression model
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

# Train and evaluate models
results = {{}}

X = df.drop(target_column, axis=1)
y = df[target_column]

# Check if the target column is not continuous
if df[target_column].dtype == 'object':
    # Convert target column to binary
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)


label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
reversed_mapping = {{str(value): key for key, value in label_mapping.items()}}

bootstrap_auc_scores = []
bootstrap_accuracy_scores = []
bootstrap_log_likelihoods = []
cnt = 1
for bootstrap_iter in range(500):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)  # Example split, adjust as needed

    # Preprocess data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    pipeline = Pipeline(steps=[('model', model)])

    grid_search = GridSearchCV(pipeline, {{}}, cv=5)
    grid_search.fit(X_train_preprocessed, y_train)

    y_pred_proba = grid_search.predict_proba(X_test_preprocessed)
    if len(np.unique(y)) == 2:  # Calculate AUC only if binary classification
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        bootstrap_auc_scores.append(auc)

    y_pred = grid_search.predict(X_test_preprocessed)
    accuracy = accuracy_score(y_test, y_pred)
    bootstrap_accuracy_scores.append(accuracy)
    # Calculate log-likelihood
    log_likelihood = np.sum(np.log(y_pred_proba[np.arange(y_test.shape[0]), y_test]))
    bootstrap_log_likelihoods.append(log_likelihood)


mean_auc = np.mean(bootstrap_auc_scores) if bootstrap_auc_scores else 'N/A'
mean_accuracy = np.mean(bootstrap_accuracy_scores)
mean_log_likelihood = np.mean(bootstrap_log_likelihoods)
results['Multinomial Logistic Regression'] = {{
    'Mean AUC': mean_auc,
    'Mean Accuracy': mean_accuracy,
    'Mean Log-Likelihood': mean_log_likelihood,
    'Actual': y_test.tolist(),
    'Predicted': y_pred.tolist(),
    'X_test': X_test.to_dict(orient='records')
}}


coef_dict = {{}}
for coef, feat in zip(grid_search.best_estimator_.named_steps['model'].coef_[0], X.columns):
    coef_dict[feat] = coef
intercept = grid_search.best_estimator_.named_steps['model'].intercept_

X_train_preprocessed_sm = sm.add_constant(X_train_preprocessed)
model_sm = sm.MNLogit(y_train, X_train_preprocessed_sm).fit()

# Get p-values
p_values = model_sm.pvalues
p_values_dict = {{}}
for p_value, feat in zip(p_values.tolist(), X.columns.tolist()):
    p_values_dict[feat] = p_value


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

results['Multinomial Logistic Regression']['Coefficients'] = coef_dict
results['Multinomial Logistic Regression']['Intercept'] = intercept.tolist() if hasattr(intercept, 'tolist') else intercept

# Note: Adjust the calculation of BIC and AIC to be compatible with mord
bic = -2 * mean_log_likelihood + np.log(len(df)) * len(X.columns)
aic = -2 * mean_log_likelihood + 2 * len(X.columns)


results['Multinomial Logistic Regression']['BIC'] = bic
results['Multinomial Logistic Regression']['AIC'] = aic
results['Multinomial Logistic Regression']['fpr'] = fpr.tolist()
results['Multinomial Logistic Regression']['tpr'] = tpr.tolist()
results['Multinomial Logistic Regression']['HeatMap'] = correlation_data
results['Multinomial Logistic Regression']['Decoding'] = reversed_mapping
results['Multinomial Logistic Regression']['P-values'] = p_values_dict

print(json.dumps(results, indent=4))


'''
    return prompt
