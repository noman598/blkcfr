from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import json
import statsmodels.api as sm

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

# Models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

param_grid = {
    'Logistic Regression': {}
}

# Train and evaluate models
results = {}
best_model_name = None
best_auc = 0

X = df.drop(target_column, axis=1)
y = df[target_column]
X_preprocessed = preprocessor.fit_transform(X)

# Check if the target column is not continuous
if df[target_column].dtype == 'object':
    # Convert target column to binary
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
reversed_mapping = {str(value): key for key, value in label_mapping.items()}

for model_name, model in models.items():

    bootstrap_auc_scores = []
    bootstrap_accuracy_scores = []
    cnt = 1
    for bootstrap_iter in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)  # Example split, adjust as needed

        # Preprocess data
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        pipeline = Pipeline(steps=[('model', model)])

        grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5)
        grid_search.fit(X_train_preprocessed, y_train)

        y_pred_proba = grid_search.predict_proba(X_test_preprocessed)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        bootstrap_auc_scores.append(auc)

        y_pred = grid_search.predict(X_test_preprocessed)
        accuracy = accuracy_score(y_test, y_pred)
        bootstrap_accuracy_scores.append(accuracy)
       
        
    mean_auc = np.mean(bootstrap_auc_scores)
    mean_accuracy = np.mean(bootstrap_accuracy_scores)
    results[model_name] = {
        'Mean AUC': mean_auc,
        'Mean Accuracy': mean_accuracy,
        'Actual': y_test.tolist(),
        'Predicted': y_pred.tolist(),
        'X_test': X_test.to_dict(orient='records')
    }

    # Get coefficients and intercept for logistic regression
    if model_name == 'Logistic Regression':
        coef_dict = {}
        for coef, feat in zip(grid_search.best_estimator_.named_steps['model'].coef_[0], X.columns):
            coef_dict[feat] = coef
        intercept = grid_search.best_estimator_.named_steps['model'].intercept_

        # X_train_preprocessed_sm = sm.add_constant(X_train_preprocessed)
        # model_sm = sm.MNLogit(y_train, X_train_preprocessed_sm).fit()

        # # Get p-values
        # p_values = model_sm.pvalues
        # p_values_dict = {}
        # for p_value, feat in zip(p_values.tolist(), X.columns.tolist()):
        #     p_values_dict[feat] = p_value

        results[model_name]['Coefficients'] = coef_dict
        results[model_name]['Intercept'] = intercept.tolist() if hasattr(intercept, 'tolist') else intercept

        # Calculate BIC and AIC using statsmodels
        X_train_sm = sm.add_constant(X_train_preprocessed)
        logit_model = sm.Logit(y_train, X_train_sm)
        
        result = logit_model.fit(disp=0)
        
        bic = result.bic
        aic = result.aic
        log_likelihood = result.llf
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        results[model_name]['BIC'] = bic
        results[model_name]['AIC'] = aic
        results[model_name]['Log-Likelihood'] = log_likelihood
        results[model_name]['fpr'] = fpr.tolist()
        results[model_name]['tpr'] = tpr.tolist()
        results[model_name]['HeatMap'] = correlation_data
        # results[model_name]['Decoding'] = reversed_mapping
        # results[model_name]['P-values'] = p_values_dict

print(json.dumps(results, indent=4))
