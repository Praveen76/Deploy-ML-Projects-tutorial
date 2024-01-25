import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from titanic_model.config.core import config
from titanic_model.pipeline import titanic_pipe
from titanic_model.processing.data_manager import load_dataset, save_pipeline


# Training Function: Defines a function run_training responsible for training the machine learning model. The function performs the following steps:
# Loads the training dataset using load_dataset.
# Splits the dataset into training and testing sets using train_test_split.
# Fits the machine learning pipeline (titanic_pipe) on the training data.
# Persists the trained model using save_pipeline.

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting
    titanic_pipe.fit(X_train,y_train)
    #y_pred = titanic_pipe.predict(X_test)
    #print("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)

    # persist trained model
    save_pipeline(pipeline_to_persist= titanic_pipe)
    # printing the score
    
    
# Script Execution: If the script is executed directly (not imported as a module), it calls the run_training function. 
# This block ensures that the training process is initiated when the script is run.
if __name__ == "__main__":
    run_training()
    
    
# In summary, this script is designed to read the Titanic training dataset, split it into training and testing sets, 
# train a machine learning model using the specified pipeline (titanic_pipe), and save the trained model for later use. 
# The script can be run independently to train the model.