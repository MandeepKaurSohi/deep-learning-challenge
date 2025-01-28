# Deep Learning Challenge: Alphabet Soup Charity Predictor
## Overview
The nonprofit foundation Alphabet Soup aims to maximize the success of its funded projects. To assist the foundation, this project builds a binary classifier using deep learning to predict whether an applicant will successfully use the provided funding. The dataset contains metadata about more than 34,000 organizations, which includes application types, classification categories, income amounts, and other relevant features.

## This project is structured into four key steps:

1.Data Preprocessing: Preparing the dataset for analysis by cleaning and encoding categorical variables, scaling numerical features, and splitting the data.
2.Compile, Train, and Evaluate the Model: Designing a deep learning model using TensorFlow and Keras, compiling it, and evaluating its performance.
3.Optimize the Model: Improving the model by adjusting features, neurons, layers, and training parameters to achieve higher accuracy.
4.Analysis Report: Summarizing results, performance, and recommendations for alternative models.
## Dataset Description
The dataset contains the following columns:

EIN and NAME: Identification columns (removed during preprocessing).
APPLICATION_TYPE: Type of application submitted to Alphabet Soup.
AFFILIATION: Affiliated sector of the applicant.
CLASSIFICATION: Government classification of the organization.
USE_CASE: Purpose of the funding request.
ORGANIZATION: Type of organization (e.g., Trust, Co-operative).
STATUS: Active status of the application.
INCOME_AMT: Income classification of the applicant.
SPECIAL_CONSIDERATIONS: Any special considerations for the application.
ASK_AMT: Funding amount requested.
IS_SUCCESSFUL: Target variable indicating if the funding was used successfully.
##  Files
The repository contains the following files:

1.charity_data.csv: The dataset used for training and testing the model.
2.AlphabetSoupCharity.ipynb: Jupyter Notebook for preprocessing, compiling, training, and evaluating the first neural network model.
3.AlphabetSoupCharity_Optimization.ipynb: Jupyter Notebook with optimizations for improving model performance.
4.AlphabetSoupCharity.h5: HDF5 file containing the trained model from the first attempt.
5.AlphabetSoupCharity_Optimization.h5: HDF5 file containing the trained model from the optimized attempt.
6.README.md: Documentation for the project.
7.Report.md: A detailed analysis of the model's performance, preprocessing steps, and recommendations.
## Steps in the Analysis
1. Data Preprocessing
Target Variable: IS_SUCCESSFUL (binary classification: 1 = successful, 0 = unsuccessful).
Features: Relevant columns such as APPLICATION_TYPE, CLASSIFICATION, USE_CASE, INCOME_AMT, and ASK_AMT.
Removed Columns: Non-beneficial identifiers like EIN and NAME. In the optimization step, STATUS, SPECIAL_CONSIDERATIONS, and ASK_AMT were also removed.
Transformations:
Rare categories in APPLICATION_TYPE and CLASSIFICATION were grouped into an "Other" category.
Categorical variables were one-hot encoded.
Numerical variables were scaled using StandardScaler.
2. Compile, Train, and Evaluate the Model
First Attempt:
Architecture:
Input Layer: 43 features.
Hidden Layers:
Layer 1: 80 neurons, ReLU activation.
Layer 2: 30 neurons, ReLU activation.
Output Layer: 1 neuron, sigmoid activation.
Performance:
Training Accuracy: ~73.6%.
Test Accuracy: ~72.63%.
Test Loss: 0.558.
Challenges: The model failed to achieve the target accuracy of 75%, indicating room for improvement.
Second Attempt:
Architecture:
Input Layer: 30 features (after further preprocessing).
Hidden Layers:
Layer 1: 10 neurons, ReLU activation.
Layer 2: 10 neurons, ReLU activation.
Output Layer: 1 neuron, sigmoid activation.
Performance:
Training Accuracy: ~80.8%.
Test Accuracy: ~78.86%.
Test Loss: 0.473.
Key Improvements:
Reduced the number of neurons and features.
Applied stricter binning thresholds for categorical features.
3. Optimize the Model
To improve accuracy, the following optimizations were made:

Dropped additional non-beneficial columns (STATUS, SPECIAL_CONSIDERATIONS, and ASK_AMT).
Reduced the number of neurons and layers to prevent overfitting.
Adjusted binning thresholds for categorical variables to reduce noise.
Increased training epochs and used a validation split during training.
4. Exporting the Model
Both models were saved as HDF5 files:

AlphabetSoupCharity.h5: First attempt.
AlphabetSoupCharity_Optimization.h5: Optimized model.
## Results and Recommendations
Performance Summary
First Attempt: Achieved 72.63% accuracy, failing to meet the target of 75%.
Second Attempt: Achieved 78.86% accuracy, surpassing the target, with reduced loss.
## Recommendations for Alternative Models
While the optimized neural network demonstrated improved performance, the following models may offer even better results for this tabular dataset:

Random Forest or Gradient Boosting (e.g., XGBoost):
Handles categorical variables effectively.
Provides feature importance insights.
Logistic Regression:
Simpler and interpretable model with potentially similar performance for binary classification tasks.
## How to Use This Project
Clone this repository to your local machine:
bash
Copy
Edit
git clone https://github.com/MandeepKaurSohi/deep-learning-challenge.git
Install the required dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Open and run the Jupyter Notebooks:
AlphabetSoupCharity.ipynb for the initial model.
AlphabetSoupCharity_Optimization.ipynb for the optimized model.
Review the exported HDF5 files for the trained models.
## Technologies Used
Python
Pandas
TensorFlow/Keras
Scikit-learn
Jupyter Notebook
Google colab

