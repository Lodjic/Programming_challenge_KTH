# Author : Loïc Thiriet

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn import decomposition


def preprocessing_trainset(normalizing, drops=None):
    y_replacement = {"Dragspel": 0, "Nyckelharpa": 1, "Serpent": 2}
    normalizing_dict = {}

    # Importing the trainingset
    df = pd.read_csv('Trainset.csv', sep=',')

    # Handling weird values on 'y'
    df["y"].replace({"ragspel": "Dragspel", "yckelharpa": "Nyckelharpa", "erpent": "Serpent"}, inplace=True)

    # Handling weird values on 'x1'
    df.loc[df.index[df["x1"] == "?"][0], ["x1"]] = np.nan   # '?' at line 716 -> df.loc[716, ["y"]] = Nyckelharpa
    df = df.drop(labels=df.index[df["x1"].isna()][0], axis=0)
    df.loc[df.index[df["x1"] == "10000000000000000000.06678"][0], ["x1"]] = 1.06678   # 1e+19 at line 301 -> df.loc[301, ["y"]] = Serpent  => might be an introduction of nineteen 0 and real value might be: 1.06678
    df["x1"] = pd.to_numeric(df["x1"])   

    # Handling weird values on 'x2'
    df.loc[df.index[df["x2"] == -1.217e+22][0], ["x2"]] = -1217.19246   # -1.217e+22 at line 441 -> df.loc[441, ["y"]] = Nyckelharpa  => might be an introduction of nineteen 0 and real value might be: -1217.19246
    if normalizing["x2"]:
        df["x2"] = (df["x2"] - df["x2"].mean()) / df["x2"].std()   # Normalization ? -> every values is between -3289.594720 and 3057.646000 looking like normal distrib with mean at 7.3
        normalizing_dict["x2"] = [df["x2"].mean(), df["x2"].std()]

    # Handling weird values on 'x3'
    df = df.drop(labels=df.index[df["x3"].isna()][0], axis=0)   # line 732 is full of nan starting from column x3 until column x13
    if normalizing["x3"]:
        df["x3"] = (df["x3"] - df["x3"].mean()) / df["x3"].std()   # Normalization ? -> every values is between 1000.043420 and 1001.275600 looking like normal distrib with mean at 1000.60
        normalizing_dict["x3"] = [df["x3"].mean(), df["x3"].std()]

    # Handling weird values on 'x4'
    if normalizing["x4"]:
        df["x4"] = (df["x4"] - df["x4"].mean()) / df["x4"].std()   # Normalization ? -> every values is between -444.088220 and 179.128650 looking like normal distrib with mean at -104
        normalizing_dict["x4"] = [df["x4"].mean(), df["x4"].std()]

    # Handling weird values on 'x5'
    # Column 'x5' is only composed of 0 except eight 1e-5 and thre -1e-5 => dropping column 'x5'
    df = df.drop(columns="x5")

    # Handling weird values on 'x6'
    df.loc[df.index[df["x6"] == "Ostra stationen"][0], "x6"] = "Östra stationen"   # 'Ostra stationen' at line 600 instead of 'Östra stationen'
    # df[df["y"] == "Dragspel"]["x6"].hist()
    # df[df["y"] == "Nyckelharpa"]["x6"].hist()
    # df[df["y"] == "Serpent"]["x6"].hist()
    df = df.drop(columns="x6")   # drop column 'x6' because 23% of na

    # Handling weird values on 'x7'
    if normalizing["x7"]:
        df["x7"] = (df["x7"] - df["x7"].mean()) / df["x7"].std()   # Normalization ? -> every values is between -4.153960 and 6.058590 looking like normal distrib with mean at 1.13
        normalizing_dict["x7"] = [df["x7"].mean(), df["x7"].std()]

    # Handling weird values on 'x8'
    if normalizing["x8"]:
        df["x8"] = (df["x8"] - df["x8"].mean()) / df["x8"].std()   # Normalization ? -> every values is between -6.388440 and 3.575680 looking like normal distrib with mean at -1.06
        normalizing_dict["x8"] = [df["x8"].mean(), df["x8"].std()]

    # Handling weird values on 'x9'
    if normalizing["x9"]:
        df["x9"] = (df["x9"] - df["x9"].mean()) / df["x9"].std()   # Normalization ? -> every values is between 5469.507990 and 5481.377140 looking more or less like normal distrib with mean at 5474.9
        normalizing_dict["x9"] = [df["x9"].mean(), df["x9"].std()]

    # Handling weird values on 'x10'
    if normalizing["x10"]:
        df["x10"] = (df["x10"] - df["x10"].mean()) / df["x10"].std()   # Normalization ? -> every values is between -89190.769700 and -89181.768570 looking like normal distrib with mean at -89186
        normalizing_dict["x10"] = [df["x10"].mean(), df["x10"].std()]

    # Handling weird values on 'x11'
    df.loc[df.index[df["x11"] == "Tru"][0], "x11"] = "True"   # 'Tru' at line 362 instead of 'True'
    df.loc[df.index[df["x11"] == "F"][0], "x11"] = "False"   # 'F' at line 674 instead of 'False'
    df.loc[df["x11"] == "True", "x11"] = "1"   # converting 'True' in '1'
    df.loc[df["x11"] == "False", "x11"] = "0"   #  converting 'False' in '0'
    df["x11"] = pd.to_numeric(df["x11"])
    if drops is not None and drops["x11"]:
        df = df.drop(columns="x11")   # drop column 'x11' because proportions of the 3 classes almost do not change between True and False

    # Handling weird values on 'x12'
    df.loc[df.index[df["x12"] == "Flase"][0], "x12"] = "False"   # 'Flase' at line 352 instead of 'False'
    df.loc[df.index[df["x12"] == "F"][0], "x12"] = "False"   # 'F' at line 674 instead of 'False'
    df.loc[df["x12"] == "True", "x12"] = "1"   # converting 'True' in '1'
    df.loc[df["x12"] == "False", "x12"] = "0"   #  converting 'False' in '0'
    df["x12"] = pd.to_numeric(df["x12"])
    if drops is not None and drops["x12"]:
        df = df.drop(columns="x12")   # drop column 'x12' because only 3,3% of True

    # Handling weird values on 'x13'
    if normalizing["x13"]:
        df["x13"] = (df["x13"] - df["x13"].mean()) / df["x13"].std()   # Normalization ? -> every values is between -8.892120 and 3.578520 looking like normal distrib with mean at -2.1
        normalizing_dict["x13"] = [df["x13"].mean(), df["x13"].std()]

    df["y"].replace(y_replacement, inplace=True)

    return df, normalizing_dict

def fetch_dataset(normalizing, drops=None):
    df, normalizing_dict = preprocessing_trainset(normalizing, drops)
    X, y = df.drop(columns="y").to_numpy(), df["y"].to_numpy()
    return X, y


def cross_validation(models, X , y, statistical_repeats=5):
    stratified_spliter = StratifiedKFold(n_splits=5, shuffle=True)
    shuffled_stratified_spliter = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
    insights = {}
    for model_name, model in models.items():
        insights[model_name] = {}
        for n in range(statistical_repeats):
            insights[model_name][f"Stratified-{n}"] = cross_val_score(model, X, y, cv=stratified_spliter)
            insights[model_name][f"Shuffled stratified-{n}"] = cross_val_score(model, X, y, cv=shuffled_stratified_spliter)
    return insights


def test_classifier(classifier, normalizing, drops=None, pcadim=0, test_size=0.2, ntrials=100):

    X, y = fetch_dataset(normalizing, drops)

    means = np.zeros(ntrials,)

    for trial in range(ntrials):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)

        # Do PCA replace default value if user provides it
        if pcadim > 0:
            pca = decomposition.PCA(n_components=pcadim)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

        # Train
        classifier.fit(X_train, y_train)
        # Predict
        accuracy = classifier.score(X_test, y_test)

        # Compute classification error
        if trial % 10 == 0:
            print("Trial:",trial,"Accuracy","%.3g" % (100*accuracy))

        means[trial] = 100 * accuracy

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "±", "%.3g" % (np.std(means)))




def preprocessing_testset(normalizing_dict, drops=None):
    y_replacement = {0: "Dragspel", 1: "Nyckelharpa", 2: "Serpent"}

    # Importing the trainingset
    df = pd.read_csv('Evaluationset.csv', sep=',')

    # Handling weird values on 'x1'
    df["x1"] = pd.to_numeric(df["x1"])   

    # Handling weird values on 'x2'
    if "x2" in normalizing_dict.keys():
        df["x2"] = (df["x2"] - normalizing_dict["x2"][0]) / normalizing_dict["x2"][1]   # Normalization if done on trainset

    # Handling weird values on 'x3'
    if "x3" in normalizing_dict.keys():
        df["x3"] = (df["x3"] - normalizing_dict["x3"][0]) / normalizing_dict["x3"][1]   # Normalization if done on trainset

    # Handling weird values on 'x4'
    if "x4" in normalizing_dict.keys():
        df["x4"] = (df["x4"] - normalizing_dict["x4"][0]) / normalizing_dict["x4"][1]   # Normalization if done on trainset

    # Handling weird values on 'x5'
    # Column 'x5' is only composed of 0 except eight 1e-5 and thre -1e-5 => dropping column 'x5'
    df = df.drop(columns="x5")

    # Handling weird values on 'x6'
    df = df.drop(columns="x6")   # drop column 'x6' because 23% of na in trainset

    # Handling weird values on 'x7'
    if "x7" in normalizing_dict.keys():
        df["x7"] = (df["x7"] - normalizing_dict["x7"][0]) / normalizing_dict["x7"][1]   # Normalization if done on trainset

    # Handling weird values on 'x8'
    if "x8" in normalizing_dict.keys():
        df["x8"] = (df["x8"] - normalizing_dict["x8"][0]) / normalizing_dict["x8"][1]   # Normalization if done on trainset

    # Handling weird values on 'x9'
    if "x9" in normalizing_dict.keys():
        df["x9"] = (df["x9"] - normalizing_dict["x9"][0]) / normalizing_dict["x9"][1]   # Normalization if done on trainset

    # Handling weird values on 'x10'
    if "x10" in normalizing_dict.keys():
        df["x10"] = (df["x10"] - normalizing_dict["x10"][0]) / normalizing_dict["x10"][1]   # Normalization if done on trainset

    # Handling weird values on 'x11'
    df.loc[df["x11"] == "True", "x11"] = "1"   # converting 'True' in '1'
    df.loc[df["x11"] == "False", "x11"] = "0"   #  converting 'False' in '0'
    df["x11"] = pd.to_numeric(df["x11"])
    if drops is not None and drops["x11"]:
        df = df.drop(columns="x11")   # drop column 'x11' because proportions of the 3 classes almost do not change between True and False

    # Handling weird values on 'x12'
    df.loc[df["x12"] == "True", "x12"] = "1"   # converting 'True' in '1'
    df.loc[df["x12"] == "False", "x12"] = "0"   #  converting 'False' in '0'
    df["x12"] = pd.to_numeric(df["x12"])
    if drops is not None and drops["x12"]:
        df = df.drop(columns="x12")   # drop column 'x12' because only 3,3% of True

    # Handling weird values on 'x13'
    if "x13" in normalizing_dict.keys():
        df["x13"] = (df["x13"] - normalizing_dict["x13"][0]) / normalizing_dict["x13"][1]   # Normalization if done on trainset

    return df, y_replacement