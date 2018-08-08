import os
from math import exp

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def dataLoading():
    if os.path.exists("../data"):
        train = pd.read_csv("../data/train.csv")
        test = pd.read_csv("../data/test.csv")
        return train, test

def data_preprocessing(train=None, test=None):
    if train is not None and test is not None:
        # 성별 변수 수치화
        train.loc[train["Sex"] == "male", "Sex"] = 0
        train.loc[train["Sex"] == "female", "Sex"] = 1

        test.loc[test["Sex"] == "male", "Sex"] = 0
        test.loc[test["Sex"] == "female", "Sex"] = 1

        train.loc[train["Embarked"] == "C", "Embarked"] = 0
        train.loc[train["Embarked"] == "S", "Embarked"] = 1
        train.loc[train["Embarked"] == "Q", "Embarked"] = 2

        test.loc[test["Embarked"] == "C", "Embarked"] = 0
        test.loc[test["Embarked"] == "S", "Embarked"] = 1
        test.loc[test["Embarked"] == "Q", "Embarked"] = 2

        train.loc[pd.isnull(train["Embarked"]), "Embarked"] = 1
        test.loc[pd.isnull(test["Embarked"]), "Embarked"] = 1

        mean_fare = train["Fare"].mean()
        test.loc[pd.isnull(test["Fare"]), "Fare"] = mean_fare

        # train.loc[train["Embarked"] == "C", "Embarked"] = np.array([0,0,1])
        # train.loc[train["Embarked"] == "S", "Embarked"] = np.array([0,1,0])
        # train.loc[train["Embarked"] == "Q", "Embarked"] = np.array([1,0,0])

    else:
        data = dataLoading()
        return data

def parameter_search_SVM(training_data = None):
    if training_data is not None:
        # 파라미터 변경은 여기서 할 것!!!
        c_list = list()
        g_list = list()
        for x in np.arange(-3, 3, 1):
            c_list.append(10.0 ** x)
        for x in np.arange(-10, 0, 1):
            g_list.append(exp(x))
        C_grid = c_list
        gamma_grid = g_list

        parameters = {'C' : C_grid, 'gamma' : gamma_grid}
        model = GridSearchCV(SVC(kernel='rbf'), parameters, cv=10)
        model.fit(training_data[0], training_data[1])

        best_C = model.best_params_['C']
        best_gamma = model.best_params_["gamma"]

        return best_C, best_gamma

def fit_over_SVM(training_data, parameters):
    model_SVM = SVC(C=parameters[0], gamma=parameters[1])
    model_SVM.fit(training_data[0], training_data[1])

    return model_SVM

def prediction(model=None, test_data=None):
    # 생성된 모델과 테스트 데이터를 이용하여 결과파일 작성

    prediction = model.predict(test_data)

    # # 테스트 (예측)
    submission = pd.read_csv("../data/gender_submission.csv", index_col="PassengerId")
    submission["Survived"] = prediction
    # print(submission.shape, type(submission))

    result_file = "../result/result_SVM.csv"
    submission.to_csv(result_file, mode='w')