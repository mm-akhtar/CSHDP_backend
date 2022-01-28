# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 20:45:44 2022

@author: kkakh
"""
# install flask

from flask import Flask, jsonify, request
from flask_cors import CORS
import decision_tree, knn, lr, naive_bayes, random_forest, svm, xg_boost
app = Flask(__name__)
CORS(app)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route('/get_prediction_results', methods=['GET'])
def get_prediction_results () :
    NB = naive_bayes.Naive_Bayes()
    DT = decision_tree.Dt()
    KNN = knn.Knn()
    LR = lr.Lr()
    RF = random_forest.Rf()
    SVM = svm.Svm()
    XG = xg_boost.Xgb()
    response = {'nb': NB.Get_Result(),
                'dt': DT.Get_Result(),
                'knn': KNN.Get_Result(),
                'lr': LR.Get_Result(),
                'rf': RF.Get_Result(),
                'svm': SVM.Get_Result(),
                'xg': XG.Get_Result()
                }
    return jsonify(response), 200

@app.route('/predict', methods=['POST'])
def predict():
    json = request.get_json()
    RF = random_forest.Rf()
    result = RF.predict(json['attributes'])

    response = {
        'result': str(result['PCA'])
        }
    return response, 200


app.run(host='localhost', port=5000)

    