
def mixe_model_code(new_table, target, threshold, random_effect, fixed_slope, fixed_coefficient, random_slope, random_coefficient, fixed_slope_value, fixed_coefficient_value, random_slope_value, random_coefficient_value):

    code = f'''

import pandas as pd
from pymongo import MongoClient
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import resample
import json
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")

# Connect to MongoDB
client = MongoClient('mongodb+srv://blackcofferconsulting:Td75gzYeUPToDAPj@cluster0.dv4rmam.mongodb.net/?retryWrites=true&w=majority')
db = client['test']
collection = db['{new_table}']

# Fetch the data from the MongoDB collection
data = pd.DataFrame(list(collection.find()))

# Drop the '_id' column as it's not needed for modeling
data.drop(columns=['_id'], inplace=True)

# Convert categorical variables to numerical codes
for column in data.select_dtypes(include=['object', 'category']):
    data[column] = LabelEncoder().fit_transform(data[column])

# Define the response variable and the random effect
response_var = '{target}'
random_effect = '{random_effect}'


# Get the list of fixed effect predictors (excluding the response variable and random effect)
# fixed_effects = [col for col in data.columns if col != response_var and col != random_effect]


fixed_slope = {fixed_slope}
fixed_coefficient = {fixed_coefficient}

random_slope = {random_slope}
random_coefficient = {random_coefficient}

fixed_values = []

if fixed_slope and fixed_coefficient:
    fixed_values.append('{fixed_slope_value}')
    fixed_values.append('{fixed_coefficient_value}')

elif fixed_slope:
    fixed_values.append('{fixed_slope_value}')
    
elif fixed_coefficient:
    fixed_values.append('{fixed_coefficient_value}')


random_values = []

if random_slope and random_coefficient:
    random_values.append('{random_slope_value}')
    random_values.append( '{random_coefficient_value}')

elif random_slope:
    random_values.append('{random_slope_value}')

elif random_coefficient:
    random_values.append('{random_coefficient_value}')



# Create the formula dynamically based on user input
if fixed_slope or fixed_coefficient:
    formula = f"{{response_var}} ~ {{' + '.join(fixed_values)}}"
else:
    formula = f"{{response_var}} ~ 1"

# Define the random effects formula dynamically
if random_slope or random_coefficient:
    random_effects_formula = f"1 + {{' + '.join(random_values)}}"
else:
    random_effects_formula = "1"



# Define the model using the dynamically created formula and random effects formula
model = mixedlm(formula, data, groups=data[random_effect], re_formula=random_effects_formula)

# Fit the model
result = model.fit()

# Get the fixed effect coefficients
fixed_coefficients = result.fe_params


random_coefficients = result.random_effects

# Get the log-likelihood of the model
log_likelihood = result.llf

# Get the number of parameters in the model
num_params = result.params.shape[0]

# Calculate AIC
aic = -2 * log_likelihood + 2 * num_params

# Calculate BIC
bic = -2 * log_likelihood + num_params * np.log(data.shape[0])

# Calculate Accuracy
predicted_values = result.predict(data[fixed_values])
actual_values = data[response_var]  # Use the binary encoded target column
accuracy = np.mean((predicted_values.round() == actual_values).astype(int))

# Calculate AUC for binary classification
auc_binary = roc_auc_score(actual_values, predicted_values)

# Get the log likelihood of the full model
log_likelihood_full = result.llf


# Define the number of bootstrap iterations
n_bootstrap = 20

# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store AUC scores from each fold and bootstrap iteration
combined_auc_scores = []

# Iterate over cross-validation splits
for train_index, test_index in cv.split(data[fixed_values], data[response_var]):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    # Initialize list to store AUC scores from bootstrap iterations for this fold
    fold_auc_scores = []

    # Iterate over bootstrap iterations
    for i in range(n_bootstrap):
        # Resample the training data with replacement
        boot_train_data = resample(train_data, replace=True, random_state=i)

        # Fit the model on the bootstrapped training data
        model = mixedlm(formula, boot_train_data, groups=boot_train_data[random_effect])
        result = model.fit()

        # Predict on the test data
        predicted_values = result.predict(test_data[fixed_values])

        # Calculate AUC for binary classification
        auc = roc_auc_score(test_data[response_var], predicted_values)
        fold_auc_scores.append(auc)

    # Calculate mean AUC score for this fold
    mean_fold_auc = np.mean(fold_auc_scores)
    combined_auc_scores.append(mean_fold_auc)

# Calculate mean and standard deviation of AUC scores across all folds
mean_combined_auc = np.mean(combined_auc_scores)
std_combined_auc = np.std(combined_auc_scores)
fpr, tpr, _ = roc_curve(test_data[response_var], predicted_values)

output = {{
    "Fixed effect coefficients": fixed_coefficients.to_dict(),
    "Random effect coefficients": {{str(occupation): {{str(term): value for term, value in coeffs.items()}} for occupation, coeffs in random_coefficients.items()}},
    "AIC": aic,
    "BIC": bic,
    "Accuracy": accuracy,
    "Mean AUC": mean_combined_auc,
    "Log likelihood": log_likelihood,
    "fpr":fpr.tolist(),
    "tpr":tpr.tolist()
}}

print(json.dumps(output, indent = 4))

'''
    return code