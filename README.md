# Deploy-ML-Projects-tutorial

Welcome to the Deploy-ML-Projects-tutorial repository! This project is designed to guide you through the process of transitioning from a research environment to a production environment for machine learning projects. The primary focus is on modularization, ensuring code testability, maintainability, and adherence to production standards. The Titanic dataset is utilized to demonstrate these concepts.

## Learning Objectives

By the end of this experiment, you will:

1. Understand the concept of modularization and convert the machine learning model developed in Jupyter Notebook into different modules tailored to specific functionalities: Data Manager, Training, Pipeline, Predict, etc.

2. Learn about testability and maintainability, dividing code into modules that are more extensible and easier to maintain and test.

3. Separate configuration from code where possible, and ensure that functionality is tested and documented.

4. Consider refactoring inefficient parts of the code base and ensure reproducibility.

5. Implement version control with clear processes for tracking releases and release versions, requirements, and dependencies.

6. Adhere to standards like PEP8 for code readability and collaboration.

7. Address scalability and performance concerns, preparing the production code for deployment to scalable infrastructure.

## Project Structure

- **config**: Configuration files for the project.
- **datasets**: Data files used for training and testing the machine learning models.
- **notebooks**: Jupyter notebooks for data exploration and analysis.
- **processing**: Scripts for data processing and feature engineering.
- **trained_models**: Saved models after training.
- **LICENSE**: MIT license terms for the project.
- **README.md**: Introduction and overview of the project.
- **VERSION**: Indicates the current version of the project.
- **pipeline.py**: Defines the machine learning pipeline for the project.
- **predict.py**: Code for making predictions using the trained models.
- **requirements.txt**: Lists dependencies and packages needed to run the project.
- **train_pipeline.py**: Code for training the machine learning models.

```
Deploy-ML-Projects-tutorial/
|-- config/
|   |-- config.yml
|   |-- __init__.py
|
|-- datasets/
|   |-- train.csv
|   |-- test.csv
|
|-- notebooks/
|   |-- Experimentation_Phase_1_Data_Exploration.ipynb
|   |-- Experimentation_Phase_2_Pipeline_Building.ipynb
|
|-- processing/
|   |-- data_management.py
|   |-- features.py
|
|-- trained_models/
|   |-- model.pkl
|   |-- scaler.pkl
|
|-- LICENSE
|-- README.md
|-- VERSION
|-- pipeline.py
|-- predict.py
|-- requirements.txt
|-- train_pipeline.py
```

## Learning Objectives (Notebooks)

### Experimentation_Phase_1_Pipeline_Building.ipynb

1. Understand and explore the data.
2. Perform data preprocessing.
3. Apply ML algorithms on the Titanic dataset.

### Experimentation_Phase_2_Pipeline_Building.ipynb

1. Create custom classes required for processing.
2. Implement the pipeline and train the model.
3. Save the trained model.

## Getting Started

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Explore the Jupyter notebooks in the `notebooks` folder to understand the data exploration and pipeline building phases.
3. Customize the project structure based on your specific needs or preferences.
4. Use the provided scripts for training, predicting, and deploying machine learning models.

## License

This repository and its contents are open-sourced under the [MIT License](LICENSE). Feel free to use, modify, and distribute these projects in accordance with the terms specified in the license.

## Issues:
If you encounter any issues or have suggestions for improvement, please open an issue in the Issues section of this repository.

## Contributing

If you have a Data Science mini-project that you'd like to share, please follow the guidelines in [CONTRIBUTING.md](https://github.com/Praveen76/Data-Science-Mini-Projects/blob/main/contributing.md).

## Code of Conduct
Please adhere to our [Code of Conduct](https://github.com/Praveen76/Data-Science-Mini-Projects/blob/main/CODE_OF_CONDUCT.md) in all your interactions with the project.

## Contact:
The code has been tested on Windows system. It should work well on other distributions but has not yet been tested. In case of any issue with installation or otherwise, please contact me on [Linkedin](https://www.linkedin.com/in/praveen-kumar-anwla-49169266/)

Happy coding!!

## **About Me**:
I’m a seasoned Data Scientist and founder of [TowardsMachineLearning.Org](https://towardsmachinelearning.org/). I've worked on various Machine Learning, NLP, and cutting-edge deep learning frameworks to solve numerous business problems.




