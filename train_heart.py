from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):
    
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x_df["Sex"] = x_df.marital.apply(lambda s: 1 if s == "M" else 0)
    
    ChestPainType = pd.get_dummies(x_df.ChestPainType, prefix="ChestPainType")
    x_df.drop("ChestPainType", inplace=True, axis=1)
    x_df = x_df.join(ChestPainType)
    
    RestingECG = pd.get_dummies(x_df.RestingECG, prefix="RestingECG")
    x_df.drop("RestingECG", inplace=True, axis=1)
    x_df = x_df.join(RestingECG)
    
    x_df["ExerciseAngina"] = x_df.marital.apply(lambda s: 1 if s == "Y" else 0)
    
    ST_Slope = pd.get_dummies(x_df.ST_Slope, prefix="ST_Slope")
    x_df.drop("ST_Slope", inplace=True, axis=1)
    x_df = x_df.join(ST_Slope)
    
    y_df = x_df.pop("HeartDisease")
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    #parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    #parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    parser.add_argument('--C', type=float, default=1.0, help="C is regularization parameter. The strength of the regularization is inversely proportional to C")
    parser.add_argument('--gamma', type=float, default=2.0, help="Gamma is kernel coefficient")
    
    # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    
    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Parameter:", np.float(args.C))
    run.log("Gamma :", np.float(args.max_iter))
    
    # HK-Step 0: I cloned the repository from the main page
    data = pd.read_csv("./heart.csv")
    df = pd.DataFrame(data)

    if not os.path.isdir('data'):
        os.mkdir('data')

    df.to_csv("data/heart.csv", index=False)
    ds2 = ws.get_default_datastore()
    ds2.upload(src_dir='./data',overwrite=True, show_progress=True)
    ds = TabularDatasetFactory.from_delimited_files(path=ds2.path('heart.csv'))
    ds = train_data.register(workspace=ws,name='heart',description='test')
    
    x, y = clean_data(ds)
    # HK-Step 2: Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    
    model = sklearn.svm.SVC(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
     
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
