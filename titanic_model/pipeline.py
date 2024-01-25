from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from titanic_model.config.core import config
from titanic_model.processing.features import embarkImputer
from titanic_model.processing.features import Mapper
from titanic_model.processing.features import age_col_tfr

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Pipeline Definition: The code defines a scikit-learn Pipeline with a sequence of named steps. Each step corresponds to a specific transformation or model in the machine learning pipeline:

# embark_imputation: Imputes missing values in the embarked column using the embarkImputer transformer.
# map_sex: Maps the values in the gender column using the Mapper transformer with predefined mappings.
# map_embarked: Maps the values in the embarked column using the Mapper transformer with predefined mappings.
# map_title: Maps the values in the title column using the Mapper transformer with predefined mappings.
# age_transform: Transforms the age column using the age_col_tfr transformer.
# scaler: Scales the features using StandardScaler.
# model_rf: Applies the RandomForestClassifier with parameters specified in the configuration.

# Each step in the pipeline is defined by a tuple with a name and the corresponding transformer or model instance. 
# The entire pipeline is stored in the variable titanic_pipe and can be used for fitting and predicting in a scikit-learn workflow. 
# This pipeline is likely intended for predicting survival outcomes in the Titanic dataset based on the provided features.

titanic_pipe = Pipeline([
    
    ("embark_imputation", embarkImputer(variables=config.model_config.embarked_var)
     ),
     ##==========Mapper======##
     ("map_sex", Mapper(config.model_config.gender_var, config.model_config.gender_mappings)
      ),
     ("map_embarked", Mapper(config.model_config.embarked_var, config.model_config.embarked_mappings )
     ),
     ("map_title", Mapper(config.model_config.title_var, config.model_config.title_mappings)
     ),
     # Transformation of age column
     ("age_transform", age_col_tfr(config.model_config.age_var)
     ),
    # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestClassifier(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth,
                                      random_state=config.model_config.random_state))
          
     ])