# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root)) # This ensures that the project's root directory is in the Python path.

from typing import Dict, List
from pydantic import BaseModel  # for creating data models
from strictyaml import YAML, load # for working with YAML files

import titanic_model

# Project Directories
PACKAGE_ROOT = Path(titanic_model.__file__).resolve().parent
print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
print("DATASET_DIR :",DATASET_DIR)
print("TRAINED_MODEL_DIR :",TRAINED_MODEL_DIR)

#Defines three Pydantic models (AppConfig, ModelConfig, and Config) to represent the configuration structure of the application. 
# These models will be used to validate the configuration loaded from the config.yml file.
class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    unused_fields: List[str]
    embarked_var: str 
    gender_var: str 
    title_var: str 
    age_var: str 
    gender_mappings: Dict[str, int]
    embarked_mappings: Dict[str, int]
    title_mappings: Dict[str, int]
  
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: int


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig

#Config File Locator: It defines a function find_config_file to locate the configuration file (config.yml). 
# If the file is found, it returns the path; otherwise, it raises an exception.
def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

#YAML Configuration Loader: It defines a function fetch_config_from_yaml to parse the YAML configuration file. 
# It takes an optional argument cfg_path (the path to the configuration file) and returns a YAML object containing the parsed configuration.
def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

#Config Validator: It defines a function create_and_validate_config to run validation on the parsed configuration values. 
# It takes an optional argument parsed_config (the parsed YAML configuration) and returns a validated Config object.
def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config

#Configuration Initialization: It calls the create_and_validate_config function and assigns the resulting Config object to the 
# variable config. This variable is likely to be used throughout the project to access various configuration settings.
config = create_and_validate_config()