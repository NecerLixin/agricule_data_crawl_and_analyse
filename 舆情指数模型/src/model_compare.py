import sklearn
import pandas as pd
import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def predict_result(model_list, train_data, train_target, test_data, test_target):
    res = []
    for model_dict in model_list:
        model = model_dict['model']
        model_name = model_dict['name']
        clf = model.fit(train_data,train_target)
        train_pred = clf.predict(train_data)
        test_pred = clf.predict(test_data)
        # print('模型: ',model_name)
        train_score = clf.score(train_data,train_target)
        test_score = clf.score(test_data,test_target)
        # print('训练集评分: ',train_score)
        # print('测试集评分: ',test_score)
        temp_dict = dict()
        temp_dict['模型'] = model_name
        temp_dict['训练集评分'] = train_score
        temp_dict['测试集评分'] = test_score
        res.append(temp_dict)
    return res
    
def get_data(file_path):
    with open(file_path,'r') as f:
        data = json.load(f)
    data_sent = []
    data_label = []
    for i in data:
        data_sent.append(i['sent'])
        data_label.append(i['label'])
    data_label = np.array(data_label)
    tf_transformer = TfidfVectorizer()
    tf_data_sent = tf_transformer.fit_transform(data_sent)
    return train_test_split(tf_data_sent,data_label,test_size=0.3,random_state=42)
def define_model():
    # 二分类模型
    # LogitsRegression, DecisionTree, RandomForest, SVM, NativeBayse,KNN
    model1 = {'name':'LogistRegression',
              'model':LogisticRegression()}
    model2 = {'name':'DecisionTree',
              'model':DecisionTreeClassifier()}
    model3 = {'name':'RandomForest',
              'model':RandomForestClassifier()}
    model4 = {'name':'SVM',
              'model':SVC()}
    model5 = {'name':'MultinomiaNB',
              'model':MultinomialNB()}
    model6 = {'name':'KNeighborsClassifier',
              'model':KNeighborsClassifier(n_jobs=2)}
    return [model1,model2,model3,model4,model5,model6]












if __name__ == '__main__':
    X_train,X_test,y_train,y_test = get_data('data/data.json')
    model_list = define_model()
    score = predict_result(model_list,X_train,y_train,X_test,y_test)
    score_df = pd.DataFrame(score)
    print(score_df)
    score_df.to_csv('mdoel_score.csv',index=None)
    
    