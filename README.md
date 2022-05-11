# Heart attach prediction

In my Capstone project, I use the Heart attack data from a Kaggle competition. With this dataset I have built a Hyperdrive and AutoML model. Furthermore, for the AutoML model, U have deployed the best model as a webservice. 

## Project Set Up and Installation

### Dataset task and access

The dataset describes Cardiovascular diseases (CVDs). Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease. (For more information see the description on the Kaggle page).

The heart attack data is in a csv format.  It was downloaded from the Kaggle page here: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction. 

We have data for each of the following features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope. Finally, in the column HeartDisease is the value that we would like to predict. The  0 and 1 predicts whether the event was a heart attack or not.


## Automated ML
First  I have ran the Automated ML. 

![](pictures/settingsautml.png)
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
