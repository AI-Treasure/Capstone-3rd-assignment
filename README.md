# Heart attack prediction

In my Capstone project, I used the Heart attack data from a Kaggle competition. With this dataset I have built a Hyperdrive and AutoML model. Furthermore, for the AutoML model, I have deployed the best model as a webservice. 

## Project Set Up and Installation

### Dataset task and access

The dataset describes Cardiovascular diseases (CVDs). Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease. (For more information see the description on the Kaggle page).

The heart attack data is in a csv format.  It was downloaded from the Kaggle page here: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction. 

We have data for each of the following features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope. Finally, in the column HeartDisease is the value that we would like to predict. The  0 and 1 predicts whether the event was a heart attack or not.


## Automated ML
First  I have ran the Automated ML with the settings below. So the experiment will time out after 30 minutes and early stopping is enabled. Note that this is my intend to get the best model ever, because otherwise you might want to experiment with different running times. The aim is merely to generate working code for classification that will set up an autoML run. 

<img src="pictures/settingsautml.png" width="300" >


### Results

The screenshots below give the results for the AutoMl. The best model from AutoML has a accuracy of 88%.

There are many ways in which this can be improved. First of all, one could experiment with the times that the model runs. Maybe results would be better if we run it for a longer time. One could also try to have higher number of cross validations. Personally, I am not always a fan of voting ensembles, so I would probably not use a voting ensemble, but just one of the models that are part of the voting ensemble and maybe play some more with their hyperparameter settings.

![](pictures/automl1.GIF)
![](pictures/automl2.GIF)
![](pictures/automl3.GIF)
![](pictures/automl4.GIF)
![](pictures/automl5.GIF)
![](pictures/automl6.GIF)
![](pictures/automl7.GIF)
![](pictures/automl8.GIF)
![](pictures/automl9.GIF)
![](pictures/automl10.GIF)

![](pictures/automl11.GIF)
![](pictures/automl12.GIF)
![](pictures/automl13.GIF)
![](pictures/automl14.GIF)
![](pictures/automl15.GIF)
![](pictures/automl16.GIF)
![](pictures/automl17.GIF)
![](pictures/automl18.GIF)
![](pictures/automl19.GIF)
![](pictures/automl20.GIF)


## Hyperparameter Tuning

For the hyperparameter tuning, I have used the RBF Support Vector Machine (the SVC model). See https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html. SKlearn has several options to do classification. And I just wanted to try out the SVM for once. This SVC model has two parameters, namely a regularization parameter C and the gamma. 

I used the following set of variables to run the model. In total I had runs, where C is 1,10 or 100 and gamma is 0.02, 0.2 or 2.
Note that before handing the data to the SVM model, we have performed one hot encoding.

### Results
The best performing of the 9 models had an accuracy of 67% for C=100 and gamma= 0.02. This is lower than the value obtained with AUTOML above.

These experiments can easily be improved. For starters, the range for which I run the model can be made much larger, with more values. But this was beyond the scope of this project. Furthermore, maybe one could perform scaling and maybe some more cleaning methods. Also it would make sense to try out different models like for instance Random Forest or other models.

![](pictures/hyper1.GIF)
![](pictures/hyper2.GIF)
![](pictures/hyper3.GIF)
![](pictures/hyper4.GIF)
![](pictures/hyper5.GIF)
![](pictures/hyper6.GIF)
![](pictures/hyper7.GIF)
![](pictures/hyper8.GIF)
![](pictures/hyper9.GIF)
![](pictures/hyper10.GIF)


## Screen Recording
The recording of my model can be found in this github repository. (I was not allowed to place it on youtube.)

