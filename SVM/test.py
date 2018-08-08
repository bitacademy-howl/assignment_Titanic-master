from SVM import data_preprocessing, parameter_search_SVM, fit_over_SVM, prediction, dataLoading
import pandas as pd

data = dataLoading()
# print(data)
data[0].loc[data[0]["Embarked"] == "C", "Embarked"] = 1.0
data[0].loc[data[0]["Embarked"] == "S", "Embarked"] = 2.0
data[0].loc[data[0]["Embarked"] == "Q", "Embarked"] = 3.0


data[1].loc[data[1]["Embarked"] == "C", "Embarked"] = 1.0
data[1].loc[data[1]["Embarked"] == "S", "Embarked"] = 2.0
data[1].loc[data[1]["Embarked"] == "Q", "Embarked"] = 3.0

print("null 포함 train: \n", data[0][pd.isna(data[0]['Embarked'])])
print("null 포함 test: \n", data[1][pd.isna(data[1]['Embarked'])])

data[0].dropna(subset=["Embarked"])
print("null 포함 train: \n", data[0][pd.isna(data[0]['Embarked'])])



# print(data[1])
# data[1].info()
# data[0].loc["Embarked"].info()
# print(data[1]["Embarked"])

# data_preprocessing(data[0], data[1]) # 데이터 processing 내에서 loading 수행
