import gzip
import struct
import joblib
import numpy as np
from flask import Flask, render_template, request, url_for
from sklearn.linear_model import Ridge
import pickle
from urllib.parse import unquote

from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)
# 피클 불러오기
ela_model = joblib.load('static/ela.pkl')
k_means_model = joblib.load('static/k_means.pkl')
lasso_model = joblib.load('static/lasso.pkl')
poly_model = joblib.load('static/poly.pkl')
rf_model = joblib.load('static/rf.pkl')
ridge_model = joblib.load('static/rid_best.pkl')


@app.route("/", methods=['GET'])
def hello():
    return render_template('index.html')


@app.route("/", methods=['GET', 'POST'])
def get_index():
    # 데이터 받아오기
    Data = unquote(request.get_data('습기'))
    Data2 = Data.replace('습기=', '').replace('일조량=','').replace('전운량=','').replace('중하층운량=','')

    Humidity_d = ''
    for i in Data2:
        if i == '&':
            Data2 = Data2[1:]
            break
        if i != '&':
            Humidity_d += i
            Data2 = Data2[1:]

    Sunshine_d = ''
    for i in Data2:
        if i == '&':
            Data2 = Data2[1:]
            break
        if i != '&':
            Sunshine_d += i
            Data2 = Data2[1:]

    Cloud_d = ''
    for i in Data2:
        if i == '&':
            Data2 = Data2[1:]
            break
        if i != '&':
            Cloud_d += i
            Data2 = Data2[1:]

    Mid_Cloud_d = ''
    for i in Data2:
        if i == '&':
            Data2 = Data2[1:]
            break
        if i != '&':
            Mid_Cloud_d += i
            Data2 = Data2[1:]

    Humidity_d = float(Humidity_d)
    Sunshine_d = float(Sunshine_d)
    Cloud_d = float(Cloud_d)
    Mid_Cloud_d = float(Mid_Cloud_d)

    print(Humidity_d, Sunshine_d, Cloud_d, Mid_Cloud_d)



    data = np.array([Humidity_d, Sunshine_d, Cloud_d, Mid_Cloud_d]).reshape(1, -1)

    poly = PolynomialFeatures(degree=2)
    # PolynomialFeatures를 이용한 다항회기 모델 학습
    X_train_poly = poly.fit_transform(data)

    k_means_prod = np.round(k_means_model.predict(X_train_poly) * 1490000, 2)
    ela_prod = np.round(ela_model.predict(X_train_poly) * 1490000, 2)
    poly_prod = np.round(poly_model.predict(X_train_poly) * 1490000, 2)
    rf_prod = np.round(rf_model.predict(X_train_poly) * 1490000, 2)
    print(type(k_means_prod))
    return render_template('index.html',
                           ela=ela_prod,
                           k_means=k_means_prod,
                           poly=poly_prod,
                           rf=rf_prod)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
