import os
import csv
import openai
import subprocess
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from pymongo import MongoClient
from gridfs import GridFS
import io 
import chardet
import base64
from io import StringIO
from io import BytesIO
from bson import ObjectId
import json
import pandas as pd 
from Logistic_prompt import Logistic_pmt
from Mixed_model_prompt import mixedmodel_pmpt
from multinominal import multinominal_pmpt
from poission_linear_prompt import poisson_linear_pmpt
from ordinal_prompt import ordinal_pmpt
from mixed_model import mixe_model_code
app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_KEY")
openai.api_key = openai_api_key

file_path = 'test.py'
file_test = 'mixed.py'
client = MongoClient('mongodb+srv://blackcofferconsulting:Td75gzYeUPToDAPj@cluster0.dv4rmam.mongodb.net/?retryWrites=true&w=majority')
db = client['test']
fs = GridFS(db, collection="uploads")



@app.route('/home', methods=['POST'])
def home():

    script_path = file_path
    output = run_python_script(script_path)
    output_dict = json.loads(output) 
    print(type(output))
    print(type(output_dict))
    print("hello")
    
    return jsonify(output_dict)

@app.route('/test', methods=['POST'])
def home2():
    payload_data = request.get_json()
    payload = payload_data.get('data',"")

    # payload = {'data'}
#     payload = {"data": [
#     {
#         "_id": "663f3abd00bf021759c336fc",
#         "projectId": "663db65c6e3c4c8e556c87b2",
#         "file_id": "663db6b9b6346b99219dac0a",
#         "templateCordinates": {
#             "x": 629.72916412353516,
#             "y": 159,
#             "_id": "663f3abd00bf021759c336fd"
#         },
#         "type": "Uploaded Single Variable Widget",
#         "data": "total_bill",
#         "__v": 0
#     },
#     {
#         "_id": "663f402febddf24f38450bd3",
#         "projectId": "663db65c6e3c4c8e556c87b2",
#         "file_id": "663db6b9b6346b99219dac0a",
#         "templateCordinates": {
#             "x": 123.7291641235352,
#             "y": 104.33333331346512,
#             "_id": "663f402febddf24f38450bd4"
#         },
#         "type": "Uploaded Group Variable Widget",
#         "data": [
#             {
#                 "label": "sex",
#                 "value": "sex"
#             },
#             {
#                 "label": "smoker",
#                 "value": "smoker"
#             },
#             {
#                 "label": "day",
#                 "value": "day"
#             },
#             {
#                 "label": "size",
#                 "value": "2"
#             }
#         ],
#         "__v": 3
#     }
# ]
# }
 
    # return jsonify({"type": str(type(payload)), "print": payload})
    max_x = float('-inf')
    max_obj = None
     
    for obj in payload:
        try:
            x_value = float(obj.get("templateCordinates", {}).get("x", float('-inf')))
            if x_value > max_x:
                max_x = x_value
                max_obj = obj
        except (TypeError, ValueError):
            # Handle cases where the x-value is missing or not a valid float
            pass 
    
    columns = []
    for item in payload:
        if 'data' in item:
            if isinstance(item['data'], str):
                columns.append(item['data'])
            else:
                for data_item in item['data']:
                    columns.append(data_item['label'])

    print(columns)


 # ---------------- identifying Mutate Variable-------------------------
    Ismutate_possible = False
    nominal = False 
    for item in payload:

        if 'levelCount' in item:
            if item['levelCount'] != 0:
                Mutate = item.get('data', None)
                if Mutate is not None:
                    # return jsonify({"levelCount": level_count, "linkVariable": link_variable})
                    # columns.append(mutate)
                    Ismutate_possible = True
                    break
        
            elif item['levelCount'] == 0 and item['subTypes'] == 'Nominal':
                Mutate = item.get('data', None)
                nominal = True
                break





    if len(columns) < 3:
        return jsonify({"result":"Insufficient data. Please add more Variables"})
 
    if max_obj:
        # Extract the required data fields
        project_id = max_obj.get('projectId')
        fileid = max_obj.get('file_id')
        single_widget = max_obj.get('type')
        other_data = max_obj.get('data')


    project_name = None
    goal_name = None
    subcategory1 = None
    threshold = None

    collection = db["projects"]
    _id = ObjectId(project_id)
    document = collection.find_one({"_id": _id})
    if document:
        project_name = document.get("projectName")
        goal_name = document.get("goalName")
        goal_type = document.get("goalType", {})
        if goal_type:
            subcategory1 = goal_type.get("typeofGoal")
        goalMetric = document.get("goalMetric", {})
        if goalMetric:
            threshold = goalMetric.get("goalMetricValue")


    # print(threshold)
    # return jsonify("hell")
    new_table = "House-price"
    target = other_data 
    file_id = ObjectId(fileid)
    # models = get_models(subcategory1)
    # models_names = ", ".join(models)

    # filter the dataset by using user speicified columns - 
    # output = filter_and_store_data(file_id, db, fs, new_table, columns)
    filter_and_store_data(file_id, db, fs, new_table, columns)
    

#--------- get the categorical columns -----------
    c_collection = db[new_table]
    # c_data = list(c_collection.find())
    sample_document = c_collection.find_one()
    # df = pd.DataFrame(c_data)
    # df = df.iloc[:, 1:]
    # print(sample_document)
    categorical_columns = []

    for key, value in sample_document.items():
        if isinstance(value, str):
            categorical_columns.append(key)

    # categorical_cols = [col for col in df.columns if df[col].dtype == 'str']

    print("Categorical col",categorical_columns)
    # return jsonify("checking")






#------------------ identify the target column properties-----------------------

    t_collection = db[new_table]

    # Retrieve distinct values in the 'loan_status' column
    distinct_loan_statuses = t_collection.distinct(target)
    if len(distinct_loan_statuses) == 2:
        status_type = 'binary'
    elif any(isinstance(value, (int, float)) for value in distinct_loan_statuses):
        status_type = 'continuous'
    else:
        status_type = 'multilevel'

    print("The 'loan_status' column is:", status_type)

    
    # to_get_best_accuracy = []

    # sample_data = t_collection.find().limit(1000)



# ----------------  Mutating the variable and deleting the old one --------------

    if Ismutate_possible:
        mutate_table = db[new_table]
        # Extracting levels from the payload
        level_data = next((item for item in payload if 'levelCount' in item), None)
        
        if level_data:
            levels = level_data.get('levels', [])
            
            projection = { '_id': 1, Mutate: 1}
            mongo_data = list(mutate_table.find({}, projection))
            for record in mongo_data:
                mutated_var = map_tip_to_level(record[Mutate], levels)
                mutate_table.update_one({'_id': record['_id']}, {'$set': {'mutated_col_name': mutated_var}})
        
            mutate_table.update_many({}, {'$unset': {Mutate: ""}})
        
            print("successfully mutated")

        # return jsonify({"message": "Data mutated and stored successfully"})



# --------------------  Nominal Mutation ----------------------------

    if nominal:
        mutate_table = db[new_table]
        nominal_data = next((item for item in payload if 'nominals' in item), None)

        if nominal_data:
            # nominals = nominal_data.get('nominalsValue', [])
            nominals = nominal_data.get('nominals', {}).get('nominalsValue', [])
            print(nominals)
            projection = {'_id': 1, Mutate: 1}
            mongo_data = list(mutate_table.find({}, projection))

            for record in mongo_data:
                mutated_var = map_nominal(record[Mutate], nominals)
                mutate_table.update_one({'_id': record['_id']}, {'$set': {'mutated_col_name': mutated_var}})

            # Delete the old column from the collection
            mutate_table.update_many({}, {'$unset': {Mutate: ""}})

            print("Successfully nominal mutation completed!")

    # return jsonify("cheking")


#---------- get the detials of dataset to send it on prompt for better result------------

    columns.remove(target)
    columns_with_plus = "+".join(columns)
    # get the schema details - 
    total_row, total_col, schema_det, categorical_cols = schema_detials(new_table)

    models_names = None
    # get the number of columns that has atleast 3-level(unique values)
    random_effect_col_list = random_effect_col(new_table)

    if target in random_effect_col_list:
        random_effect_col_list.remove(target)

    print(random_effect_col_list)
    # 
#------------------- Run the Mixed model code -----------------------


        # Check if "model" is "mixed model"
    model_mixed = False
    random_effect = None 
    for item in payload:
        if "model" in item and item["model"] == "mixed model":
            model_mixed = True
            break

    # If "model" is "mixed model", extract specific keys
    if model_mixed:
        variables = {}
        for item in payload:
            if isinstance(item, dict): 
                 for key, value in item.items(): 
                     if key in ["Fixed Slope", "Fixed Coefficient", "Random Slope", "Random Coefficient"]:
                         variables[key] = value    
        print("Variables:", variables)


    # return jsonify("checking")



    
    

    # print(fixed_slope)
    # print(fixed_coefficient)
    # print(random_coefficient)
    # return jsonify("j")

    result_dict = []

    if model_mixed == True:
        fixed_slope = False
        fixed_coefficient = False
        random_slope = False
        random_coefficient = False

        fixed_slope_value = None
        fixed_coefficient_value = None
        random_slope_value = None
        random_coefficient_value = None


        if 'Fixed Slope' in variables:
            fixed_slope = True
            fixed_slope_value = variables['Fixed Slope']

        if 'Fixed Coefficient' in variables:
            fixed_coefficient = True
            fixed_coefficient_value = variables['Fixed Coefficient']

        if 'Random Slope' in variables:
            random_slope = True
            random_slope_value = variables['Random Slope']

        if 'Random Coefficient' in variables:
            random_coefficient = True
            random_coefficient_value = variables['Random Coefficient']

            
        random_effect = random_effect_col_list[1]
        attempts = 3
        for _ in range(attempts):

            response = None

            # response = get_gpt_response(mixedmodel_pmpt(schema_det, total_row, total_col, target, new_table, goal_name, threshold, models_names, categorical_cols, columns_with_plus)) 

            # result = response[response.find('\n')+1:response.rfind('\n')]
            
            result = mixe_model_code(new_table, target, threshold, random_effect, fixed_slope, fixed_coefficient, random_slope, random_coefficient, fixed_slope_value, fixed_coefficient_value, random_slope_value, random_coefficient_value)
            with open(file_test, 'w') as f:
                print(result, file = f)

            script_path = file_test
            output = run_python_script(script_path)
            if start_index != -1:
                end_index = output.rfind('}')
                if end_index != -1:
                    json_str = output[start_index:end_index + 1]

            if 'Error:' in output:
                print("Retrying...")
            elif output:
                # print(output)
                output_dict = json.loads(json_str) 
                # output_dict = output
                # output_dict["freeze"] = True
                # return jsonify({"result": output_dict})
                result_dict.append({"Mixed Model":output_dict})
                break 
    
    # return jsonify({"result": result_dict})
        # return jsonify({"result": "create model again"})




#-------------   all models except mixed ----------------

    if status_type == 'continuous':
        prompt_name = [poisson_linear_pmpt]
        file = ['poission_linear.py']
    elif status_type == 'binary':
        prompt_name = [multinominal_pmpt, ordinal_pmpt, Logistic_pmt]
        file = ['multinomnal.py','ordinal.py', 'logistic.py']
        # prompt_name = [multinominal_pmpt]
        # file = ['multinomnal.py']
    else:
        prompt_name = [ordinal_pmpt]
        file = ['ordinal.py']
  

    # prompt_name = [multinominal_pmpt, ordinal_pmpt,Logistic_pmt, poisson_linear_pmpt]
    # file = ['multinomnal.py', 'ordinal.py', 'logistic.py','poission_linear.py']

    
    for i in range(0, len(prompt_name)):

        attempts = 3
        for _ in range(attempts):

            response = None

            response = get_gpt_response(prompt_name[i](schema_det, total_row, total_col, target, new_table, goal_name, threshold, models_names)) 

            result = response[response.find('\n')+1:response.rfind('\n')]
            with open(file[i], 'w') as f:
                print(result, file = f)
            # try:

            script_path = file[i]
            output = run_python_script(script_path)
            # print(output)
            json_str = ''
            start_index = output.find('{')
            if start_index != -1:
                end_index = output.rfind('}')
                if end_index != -1:
                    json_str = output[start_index:end_index + 1]

            # print(json_str)
            # print(type(output))
            if 'Error:' in output:
                print("Retrying...")
            else:
                output_dict = json.loads(json_str) 
                output_dict["freeze"] = True
                
  
                result_dict.append(output_dict)
                break
        
        
    if len(result_dict) == 0:
        return jsonify({"result": "create model again"})
    else:
        # result_dict["freeze"] = True
        result_dict = best_model(result_dict)
        # result_dict["categorical_column"] = categorical_columns

        collection = db["Output"]
        data2 = {
            "project_id": project_id,
            "data": result_dict
        }
        collection.insert_one(data2)
        return jsonify({"result": result_dict})




@app.route('/data', methods=['GET'])
def get_data():
    # payload = request.get_json()
    # projectid = request.json.get('project_id')
    project_id = request.args.get('project_id')
    # project_id = payload.get('project_id')
    # print(project_id)
    # p_id = ObjectId(project_id)
    # print(p_id)
    if project_id is None:
        return jsonify({"error": "Project ID is missing in the payload"}), 400
    get_collection = db["Output"]
    datavalue = get_collection.find_one({"project_id":project_id})
    
    if datavalue:
        model_ouput = datavalue.get("data")
    # if data:
        # data['_id'] = str(data['_id'])
        return jsonify({"res": model_ouput})
    else:
        return jsonify({"error": "Data not found for project_id"}), 404


@app.route('/Removed', methods=['DELETE'])
def delete_data():
    project_id = request.args.get('project_id')
    
    if project_id is None:
        return jsonify({"error": "Project ID is missing in the payload"})
    
    collection = db["Output"]
    result = collection.delete_many({"project_id": project_id})
    
    if result.deleted_count > 0:
        return jsonify({"message": f"Data associated with project ID {project_id} deleted successfully"})
    else:
        return jsonify({"error": "Data not found for project_id"})




@app.route('/mixed_model_identiying', methods=['POST'])
def mixed_model_fun():
    payload_data = request.get_json()
    payload = payload_data.get('data',"")

 
    # return jsonify({"type": str(type(payload)), "print": payload})
    max_x = float('-inf')
    max_obj = None
     
    for obj in payload:
        try:
            x_value = float(obj.get("templateCordinates", {}).get("x", float('-inf')))
            if x_value > max_x:
                max_x = x_value
                max_obj = obj
        except (TypeError, ValueError):
            # Handle cases where the x-value is missing or not a valid float
            pass 
    
    columns = []
    for item in payload:
        if 'data' in item:
            if isinstance(item['data'], str):
                columns.append(item['data'])
            else:
                for data_item in item['data']:
                    columns.append(data_item['label'])
    print(columns)

    if len(columns) < 3:
        return jsonify({"result":"Insufficient data. Please add more Variables"})
 
    if max_obj:
        # Extract the required data fields
        project_id = max_obj.get('projectId')
        fileid = max_obj.get('file_id')
        single_widget = max_obj.get('type')
        other_data = max_obj.get('data')


    project_name = None
    goal_name = None
    subcategory1 = None
    threshold = None

    collection = db["projects"]
    _id = ObjectId(project_id)
    document = collection.find_one({"_id": _id})
    if document:
        project_name = document.get("projectName")
        goal_name = document.get("goalName")
        goal_type = document.get("goalType", {})
        if goal_type:
            subcategory1 = goal_type.get("typeofGoal")
        threshold_Metric = document.get("thresholdMetric", {})
        if threshold_Metric:
            threshold = threshold_Metric.get("thresholdMetricValue")



    new_table = "House-price"
    target = other_data 
    file_id = ObjectId(fileid)
    output = filter_and_store_data(file_id, db, fs, new_table, columns)
    
    # get the number of columns that has atleast 3-level(unique values)
    random_effect_col_list = random_effect_col(new_table)

    if target in random_effect_col_list:
        random_effect_col_list.remove(target)


    if len(random_effect_col_list)>=2:
        return jsonify({"result": True})
    else:
        return jsonify({"result": False})


@app.route('/type of columns', methods=['POST'])
def types_of_col():
    payload_data = request.get_json()
    payload = payload_data.get('data',"")

        # Extract the single file_id from the payload
    file_id = ObjectId(payload[0]['file_id'])

    # Function to determine the type of a column
    def determine_column_type(values):
        if pd.api.types.is_bool_dtype(values):
            return 'boolean'
        elif pd.api.types.is_numeric_dtype(values):
            if values.nunique() > 10:  # Assuming more than 10 unique values makes it continuous
                return 'continuous'
            else:
                return 'discrete'
        elif pd.api.types.is_string_dtype(values):
            if values.map(len).max() <= 255:  # Arbitrary limit for short strings
                return 'categorical'
            else:
                return 'text'
        elif pd.api.types.is_datetime64_any_dtype(values):
            return 'datetime'
        else:
            return 'mixed'

    file_exists = db['uploads.files'].find_one({"_id": file_id})
    if not file_exists:
        print(f"File with ID {file_id} does not exist in GridFS.")
    else:
        # Retrieve the file from GridFS
        file_data = fs.get(file_id).read()

        # Load the data into a pandas DataFrame (assuming CSV format)
        data = pd.read_csv(BytesIO(file_data))

        # Initialize dictionary to store column types
        column_types = {}

        # Analyze each column in the DataFrame
        for column in data.columns:
            column_type = determine_column_type(data[column])
            column_types[column] = column_type

        # # Print the column types
        # print("File ID:", file_id)
        # for column, column_type in column_types.items():
        #     print(f"  {column}: {column_type}")

    return jsonify({"result":column_types})



# @app.route('/graph', methods=['POST'])
# def graphs():
#     payload_data = request.get_json()
#     payload = payload_data.get('data',"")

#         # Extract the single file_id from the payload
#     file_id = ObjectId(payload[0]['file_id'])

#     # Function to determine the type of a column
#     def determine_column_type(values):
#         if pd.api.types.is_bool_dtype(values):
#             return 'boolean'
#         elif pd.api.types.is_numeric_dtype(values):
#             if values.nunique() > 10:  # Assuming more than 10 unique values makes it continuous
#                 return 'continuous'
#             else:
#                 return 'discrete'
#         elif pd.api.types.is_string_dtype(values):
#             if values.map(len).max() <= 255:  # Arbitrary limit for short strings
#                 return 'categorical'
#             else:
#                 return 'text'
#         elif pd.api.types.is_datetime64_any_dtype(values):
#             return 'datetime'
#         else:
#             return 'mixed'

#     file_exists = db['uploads.files'].find_one({"_id": file_id})
#     if not file_exists:
#         print(f"File with ID {file_id} does not exist in GridFS.")
#     else:
#         # Retrieve the file from GridFS
#         file_data = fs.get(file_id).read()

#         # Load the data into a pandas DataFrame (assuming CSV format)
#         data = pd.read_csv(BytesIO(file_data))

#         # Initialize dictionary to store column types
#         column_types = {}

#         # Analyze each column in the DataFrame
#         for column in data.columns:
#             column_type = determine_column_type(data[column])
#             column_types[column] = column_type

#         # # Print the column types
#         # print("File ID:", file_id)
#         # for column, column_type in column_types.items():
#         #     print(f"  {column}: {column_type}")

#     Qualitative = []
#     Quantitative = []

#     # for key, value in column_types.items():
#     #     if value in ['continuous', 'discrete']:
#     #         Quantitative.append(key)
#     #     elif 

#     return jsonify({"result":column_types})

System_Prompt = """
        Assist in training a machine learning model with a dataset stored in MongoDB , returning coefficients. Instructions include providing necessary code, focusing solely on code, establishing a connection to MongoDB using Python with provided credentials, predicting a specified target column with preprocessing for string columns, extracting model coefficients after training, specifying required libraries to import, including necessary pymongo and bson modules, io, ColumnTransformer.
        pleae import the library - from sklearn.compose import ColumnTransformer
        """

def get_gpt_response(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": System_Prompt},
                  {"role": "user", "content": message}],
        temperature=0
    )
    return response['choices'][0]['message']['content']


def run_python_script(script_path):
    try:
        # Use subprocess to run the Python script and capture its output
        result = subprocess.run(["python", script_path], check=True, stdout=subprocess.PIPE, text=True)

        # Check if the script executed successfully
        if result.returncode == 0:
            # Print the captured output of the script
            print("Script executed successfully. Output:")
            # print(result.stdout)
            return result.stdout
        else:
            # If there was an error during script execution, print error message
            # print(f"Error: Script returned non-zero exit status {result.returncode}")
            return f"Error: Script returned non-zero exit status {result.returncode}"
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return f"Error: {e}"
    except FileNotFoundError:
        # print("Error: Python interpreter not found. Ensure Python is installed.")
        return "Error: Python interpreter not found. Ensure Python is installed."





def schema_detials(new_table):

    # Replace 'House-price' with your collection name
    collection_name = db['House-price']

    # Get the first document from the collection
    sample_document = collection_name.find_one()

    # Extract column names and their types from the sample document
    column_types = {key: type(value).__name__ for key, value in sample_document.items()}

    # Get total number of documents (rows) in the collection
    total_rows = collection_name.count_documents({})

    # Get total number of columns
    total_columns = len(sample_document)

    # Convert schema details to string
    schema_details = "\n".join([f"{column}: {data_type}" for column, data_type in column_types.items()])

    categorical_columns = [column for column, data_type in column_types.items() if data_type == 'str' or data_type == 'unicode']

    return total_rows, total_columns, schema_details, categorical_columns


# def get_models(subcategory1):
#     if subcategory1 == "Quantitative":
#         models = ["Linear Regression","Poisson Regression", "Ridge Regression", "Multiple Regression"]
#     elif subcategory1 == "Qualitative":
#         models = ["Linear Regression", "Logistic Regression", "Oridinal Logistics Regression", "Multiple regression"]
#     return models


def map_nominal(value, nominals):
    for nominal in nominals:
        if value == nominal['nominalName']:
            return nominal['nominalValue']
    return None



# ------ mutate widget function---------
def map_tip_to_level(tip, levels):
    for level in levels:
        if int(level['levelMinRange']) <= tip <= int(level['levelMaxRange']):
            return int(level['levelValue'])
    return None


def random_effect_col(new_table):

    collection = db[new_table]

    # Getting the list of distinct values for each field
    unique_values = {}
    for field in collection.find_one().keys():
        unique_values[field] = collection.distinct(field)

    # Removing numerical columns and filtering columns with at least 3 unique values
    col = []
    filtered_unique_counts = {}
    for field, values in unique_values.items():
        if len(values) >= 3 and all(not isinstance(value, (int, float)) for value in values):
            # filtered_unique_counts[field] = len(values)
            col.append(field)
    return col


# get the best model after getting the output - 
def best_model(result_dict):
    best_model_name = ""
    highest_accuracy = float('inf')

    # Iterate over the list of dictionaries
    for model_dict in result_dict:
        for model_name, model_data in model_dict.items():
            if isinstance(model_data, dict):
                if "AIC" in model_data and model_data["AIC"] < highest_accuracy:
                    highest_accuracy = model_data["AIC"]
                    best_model_name = model_name

    # Append the name of the best model to the dictionary
    result_dict.append({"Best Model Name": best_model_name})

    return result_dict













# def filter_and_store_data(file_id, db, fs, new_table, columns_to_filter):
#     # Retrieve the file from GridFS
#     grid_out = fs.find_one({'_id': file_id})

#     if grid_out:
#         # Read the data from the GridFS chunks
#         data = grid_out.read()

#         # Decode the data based on its type
#         if grid_out.content_type == 'text/plain':
#             try:
#                 # Detect encoding
#                 detected_encoding = chardet.detect(data)
#                 decoded_data = data.decode(detected_encoding['encoding'])
#             except (UnicodeDecodeError, TypeError):
#                 return "Unable to decode data as text"
#         else:
#             try:
#                 # Fix base64 padding issues
#                 missing_padding = len(data) % 4
#                 if missing_padding:
#                     data += b'=' * (4 - missing_padding)
#                 decoded_data = base64.b64decode(data).decode('utf-8')
#             except (base64.binascii.Error, UnicodeDecodeError):
#                 return "Unable to decode data as base64"

#         # Convert the data from CSV format to list of dictionaries
#         reader = csv.DictReader(StringIO(decoded_data))

#         # Filter the columns based on column names
#         filtered_data = []
#         for row in reader:
#             filtered_row = {}
#             for col, val in row.items():
#                 if col in columns_to_filter:
#                     # Try converting the value back to float
#                     try:
#                         val = float(val)
#                     except ValueError:
#                         pass  # If conversion fails, keep it as it is
#                     filtered_row[col] = val
#             filtered_data.append(filtered_row)

#         if new_table in db.list_collection_names():
#             db[new_table].drop()

#         # Create a new collection to store the filtered data
#         new_collection = db[new_table]

#         # Insert the filtered data into the new collection
#         new_collection.insert_many(filtered_data)

#         return "Filtered data has been stored successfully in the new collection."
#     else:
#         return "No document found with the specified file ID."

def filter_and_store_data(file_id, db, fs, new_table, columns_to_filter):
    # Retrieve the file from GridFS
    grid_out = fs.find_one({'_id': file_id})

    if grid_out:
        # Read the data from the GridFS chunks
        data = grid_out.read().decode('utf-8')

        # Convert the data from CSV format to list of dictionaries
        reader = csv.DictReader(StringIO(data))

        # Filter the columns based on column names
        filtered_data = []
        for row in reader:
            filtered_row = {}
            for col, val in row.items():
                if col in columns_to_filter:
                    # Try converting the value back to float
                    try:
                        val = float(val)
                    except ValueError:
                        pass  # If conversion fails, keep it as it is
                    filtered_row[col] = val
            filtered_data.append(filtered_row)

        if new_table in db.list_collection_names():
            db[new_table].drop()

        # Create a new collection to store the filtered data
        new_collection = db[new_table]

        # Insert the filtered data into the new collection
        new_collection.insert_many(filtered_data)

        return "Filtered data has been stored successfully in the new collection."
    else:
        return "No document found with the specified file ID."


#host and port

if __name__ == '__main__':

    app.run(host= '0.0.0.0', port=5000)  # Run the app in debug mode







