from SVM import data_preprocessing, parameter_search_SVM, fit_over_SVM, prediction, dataLoading

import pandas as pd

# 전체 >> 데이터 로딩 > 전처리 > GridSearchCV를 이용한 최적 파라미터 search (with CV) > 모델 생성 > 예측
#      >> dataLoading > data_preprocessing > parameter_search_SVM > fit_over_SVM > prediction

# 데이터 로딩
data = dataLoading()
data_preprocessing(data[0], data[1]) # 데이터 processing 내에서 loading 수행

feature_names = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

X_train = data[0][feature_names]
Y_train = data[0]["Survived"]
X_test = data[1][feature_names]

parameters = parameter_search_SVM((X_train, Y_train))
print("SVM best C : " + str(parameters[0]))
print("SVM best gamma : " + str(parameters[1]))

model = fit_over_SVM((X_train, Y_train), parameters)
prediction(model, X_test)




# for c in c_list:
#     for g in g_list:
#         model_SVM = SVC(C=c, gamma=best_gamma)
#         model_SVM.fit(X_train_cv, Y_train_cv)
#         prediction = model_SVM.predict(X_test_cv)
#
#         accuracy = metrics.accuracy_score(Y_test_cv, prediction)
#         print('Accuracy    = ' + str(np.round(accuracy, 2)))
# ########################################################################################################################
