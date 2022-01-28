# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:51:59 2022

@author: kkakh
"""

# Decision Tree classification

# import libraries
import pandas as pd
import math

class Dt:
    def __init__(self):
        self.result= {
            'General': {
                'Precision' : '',
                'Recall': '',
                'Accuracy': '',
                'F1_Score': ''
                },
            'PCA': {
                'Precision' : '',
                'Recall': '',
                'Accuracy': '',
                'F1_Score': ''
                }
            }
        self.datset = pd.read_csv('heart_statlog_cleveland_hungary.csv')
        
    def preprocessing (self):
        dataset = self.datset
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        # spliting the dataset into training set and test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        return X_train, X_test, y_train, y_test, sc
    
    def train_predict (self, X_train, y_train):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion='entropy', random_state=3) 
        classifier.fit(X_train, y_train)
        return classifier
    
    def save_result (self, r_type, y_test, y_pred, accuracy):
        from sklearn.metrics import precision_score, recall_score
        p_score = precision_score(y_test, y_pred)*100
        r_score = recall_score(y_test, y_pred)*100
        a_score = accuracy*100
        f1_score =  ((2*(r_score * p_score))/ (r_score + p_score))
        self.result[r_type]['Precision'] = (math.ceil(p_score*100)/100)
        self.result[r_type]['Recall'] = (math.ceil(r_score*100)/100)
        self.result[r_type]['Accuracy'] = (math.ceil(a_score*100)/100)
        self.result[r_type]['F1_Score'] = (math.ceil(f1_score*100)/100)
        return
    
    def General (self):
        X_train, X_test, y_train, y_test, sc = self.preprocessing()
        classifier = self.train_predict(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator= classifier, X = X_train, y = y_train, cv = 10)
        accuracy = accuracies.mean()
        self.save_result('General', y_test, y_pred, accuracy)
        return
    
    def apply_PCA (self, X_train, X_test):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        return X_train, X_test, pca
    
    def PCA(self):
        X_train, X_test, y_train, y_test, sc = self.preprocessing()
        X_train, X_test, pca = self.apply_PCA(X_train, X_test)
        classifier = self.train_predict(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator= classifier, X = X_train, y = y_train, cv = 10)
        accuracy = accuracies.mean()
        self.save_result('PCA', y_test, y_pred, accuracy)
        return
    
    def predict_PCA(self, value):
        X_train, X_test, y_train, y_test, sc = self.preprocessing()
        X_train, X_test, pca = self.apply_PCA(X_train, X_test)
        value= pca.transform(sc.transform([value]))
        classifier = self.train_predict(X_train, y_train)
        y_pred = classifier.predict(value)
        return y_pred
    
    def predict_General (self, value):
        X_train, X_test, y_train, y_test, sc = self.preprocessing()
        value= sc.transform([value])
        classifier = self.train_predict(X_train, y_train)
        y_pred = classifier.predict(value)
        return y_pred
    
    def predict(self, value):
        predicted = {
            'General' : self.predict_General(value)[0],
            'PCA' : self.predict_PCA(value)[0]
            }
        return predicted
    
    def Get_Result (self):
        self.General()
        self.PCA()
        return self.result

'''
DT = Dt()

print(DT.Get_Result())

predict = DT.predict([37,1,4,140,207,0,0,130,1,1.5,2])

print(predict) 
'''
"""
output in %:
    {'General': 
        {'Precision': 93.05, 'Recall': 84.93, 'Accuracy': 86.56, 'F1_Score': 88.8},
    'PCA': 
        {'Precision': 88.34, 'Recall': 84.13, 'Accuracy': 88.02, 'F1_Score': 86.18}
    }
"""