import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report


if __name__ == '__main__':
    data = pd.read_csv('data/Train.csv')
    # tf-idf
    tf_transformer = TfidfVectorizer()
    tf_train_data = tf_transformer.fit_transform(data.review)
    print(tf_train_data.shape)
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(tf_train_data, data.label, test_size=0.2, random_state=42)
    # 训练
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    # 评价
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))


