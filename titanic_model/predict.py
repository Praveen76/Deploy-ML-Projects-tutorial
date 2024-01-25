from typing import Union
import pandas as pd
import numpy as np

from titanic_model import __version__ as _version
from titanic_model.config.core import config
from titanic_model.pipeline import titanic_pipe
from titanic_model.processing.data_manager import load_pipeline
from titanic_model.processing.data_manager import pre_pipeline_preparation
from titanic_model.processing.validation import validate_inputs


import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


# Load Pipeline: It loads the trained machine learning pipeline (titanic_pipe) from a saved file using the load_pipeline function.
pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
titanic_pipe= load_pipeline(file_name=pipeline_file_name)

# Prediction Function: Defines a function make_prediction that takes input data (either a DataFrame or a dictionary) and returns a dictionary containing predictions, version information, and any validation errors. The function performs the following steps:
# Validates the input data using the validate_inputs function.
# Reindexes the validated data to match the expected feature order defined in the configuration.
# If there are no validation errors, it makes predictions using the loaded pipeline (titanic_pipe).

def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = titanic_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = titanic_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

# Script Execution: If the script is executed directly (not imported as a module), it creates a sample input data (data_in) and 
# calls the make_prediction function with this input data.
if __name__ == "__main__":

    data_in={'PassengerId':[79],'Pclass':[2],'Name':["Caldwell, Master. Alden Gates"],'Sex':['male'],'Age':[0.83],
                'SibSp':[0],'Parch':[2],'Ticket':['248738'], 'Cabin':[np.nan,],'Embarked':['S'],'Fare':[29]}
    
    make_prediction(input_data=data_in)

# In summary, this script is designed to load a trained machine learning pipeline, validate input data, and make predictions using the 
# loaded pipeline. The example in the __main__ block demonstrates how to use the script to make predictions with sample input data.