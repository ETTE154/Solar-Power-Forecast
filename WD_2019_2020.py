# utf8 인코딩 설정
# -*- coding: utf-8 -*-

# 머신러닝을 위한 라이브러리 임포트
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# 경고창 제거
import warnings
warnings.filterwarnings('ignore')

# csv 파일 읽어오기
KWP_df = pd.read_csv('KWP.csv', encoding='cp949')

Sw_2019_df = pd.read_csv('WD_2019.csv', encoding='cp949')
Sw_2020_df = pd.read_csv('WD_2020.csv', encoding='cp949')

# KWP_df의 '안산연성정수장태양광' 만 추출
KWP_df = KWP_df[KWP_df['발전기명'] == '안산연성정수장태양광']

# KWP_df의 2019~2022년 데이터만 추출
KWP_df = KWP_df[KWP_df['년월일'].str.contains('2019|2020')]
# KWP_df의 콜럼의 13시 데이터만 추출
KWP_df = KWP_df[['년월일', '13시']]
KWP_df.head()
#%%
# KWP_df의 '년월일' 컬럼에서 '-' 제거
KWP_df['년월일'] = KWP_df['년월일'].str.replace('-', '')
# KWP_df의 '년월일'을 'date'로 컬럼명 변경
KWP_df.rename(columns={'년월일': 'date'}, inplace=True)
# KWP_df 13시 데이터를 'G_Value'로 컬럼명 변경
KWP_df.rename(columns={'13시': 'G_Value'}, inplace=True)

# Sw_2019_df, Sw_2020_df 합치기
Sw_df1 = pd.concat([Sw_2019_df, Sw_2020_df])
Sw_df1.head()

# Sw_df의 13:00 데이터만 추출
Sw_df = Sw_df1[Sw_df1['일시'].str.contains('13:00')]
Sw_df.head()

Sw_df.drop(['지점명', '지점','5cm 지중온도(°C)','풍향(16방위)'], axis=1, inplace=True)
Sw_df.head()
#%%
# Sw_df1의 '일시' 컬럼에서 '-', '13:00' 제거
Sw_df['일시'] = Sw_df['일시'].str.replace('-', '')
Sw_df['일시'] = Sw_df['일시'].str.replace('13:00', '')
# Sw_df1의 '일시'을 'date'로 컬럼명 변경
Sw_df.rename(columns={'일시': 'date'}, inplace=True)
Sw_df.head()

# SW_df의 칼럼명을 영어로 변경
Sw_df.rename(columns={'일시': 'date', '풍속(m/s)': 'Wind_Speed', '습도(%)': 'Humidity', '기온(°C)': 'Temperature', '일조(hr)': 'Sunshine', '일사(MJ/m2)': 'Solar_Radiation', '현지기압': 'local_pressure', '전운량(10분위)': 'Cloud_Cover', '지면온도(°C)': 'Ground_Temperature', '중하층운량(10분위)':'lower middle cloud'}, inplace=True)
Sw_df.head()

# Sw_df의 증기압, 현지가압, 최고운고 영어로 변경
Sw_df.rename(columns={'증기압(hPa)': 'Vapor_Pressure', '현지기압(hPa)': 'local_pressure', '최고운고(m)': 'Highest_cloud'}, inplace=True)
Sw_df.head()

# Sw_df의 최저운고 영어로 변경
Sw_df.rename(columns={'최저운고(100m )': 'Lowest_cloud'}, inplace=True)
Sw_df.head()
#%%
# Sw_df의 Nan값을 0으로 변경
Sw_df.fillna(0, inplace=True)
Sw_df.head()
# %%
# Sw_df의 date를 인덱스로 설정
Sw_df.set_index('date', inplace=True)
# 오름차순 정렬
Sw_df.sort_index(inplace=True)
Sw_df.head()
# %%
KWP_df.set_index('date', inplace=True)
# 오름차순 정렬
KWP_df.sort_index(inplace=True)
KWP_df.head()
#%%
Sw_df.columns
# %%
# KWP_df, Sw_df 인덱스 int형으로 변경
KWP_df.index = KWP_df.index.astype(int)
Sw_df.index = Sw_df.index.astype(int)
# Kwp_df, Sw_df 병합
KWP_Sw_df = pd.merge(KWP_df, Sw_df, left_index=True, right_index=True)
KWP_Sw_df.head()
# %%
KWP_Sw_df.tail()
# %%
# Kwpsw_df 히트맵
plt.figure(figsize=(20, 20))
sns.heatmap(KWP_Sw_df.corr(), annot=True, fmt='.2f', cmap='Blues')
plt.show()
# %%
# KWP_Sw_df의 칼럼에서 G_Value와의 관계도가 1, -1에 가까운 칼럼만 추출
KWP_Sw_df1 = KWP_Sw_df[['G_Value','Humidity','Sunshine','Cloud_Cover', 'lower middle cloud']]   
KWP_Sw_df1.head()
# %%
# KWP_Sw_df1 히트맵
plt.figure(figsize=(20, 20))
sns.heatmap(KWP_Sw_df1.corr(), annot=True, fmt='.2f', cmap='Blues')
plt.show()
# %%
#%%
# G_Value 값을 1490000으로 나누어 정규화
KWP_Sw_df1['G_Value'] = KWP_Sw_df1['G_Value'] / 1490000
KWP_Sw_df1.head()
# %%
# 이상치 제거
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(KWP_Sw_df1))
print(z)
#%%
# 이상치 제거
KWP_Sw_df1 = KWP_Sw_df1[(z < 3).all(axis=1)]
KWP_Sw_df1.head()
# %%
# %%
# df X, y 분리
X = KWP_Sw_df1.drop('G_Value', axis=1)
y = KWP_Sw_df1['G_Value']
#%%
# 스탠다드 스케일러를 이용한 정규화
from sklearn.preprocessing import StandardScaler
    
# %%
# 다항회귀분석
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# %%
# 머신러닝을 위한 Xtrain, Xtest, ytrain, ytest 분리
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)
#%%
# Xtrain, Xtest, ytrain, ytest 정보 확인
print(Xtrain.shape)
print(ytrain.shape)
print(Xtest.shape)
print(ytest.shape)
# %%
# PolynomialFeatures를 이용한 다항회기 모델 학습
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
#%%
# 2차 다항회귀 모델
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(Xtrain, ytrain)
poly = PolynomialFeatures(degree=2)
# PolynomialFeatures를 이용한 다항회기 모델 학습
X_train_poly = poly.fit_transform(Xtrain)
X_test_poly = poly.fit_transform(Xtest)
# %%
# 다항회귀 모델의 학습
model.fit(X_train_poly, ytrain)
# %%
# 학습된 모델의 예측값
y_pred = model.predict(X_test_poly)

# %%
# 다항회기 평가
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# %%
# MSE, RMSE, R2, score
poly_mse = mean_squared_error(ytest, y_pred)
poly_rmse = np.sqrt(mean_squared_error(ytest, y_pred))
poly_r2 = r2_score(ytest, y_pred)
poly_score = model.score(X_test_poly, ytest)

# MSE, RMSE, R2, score 소수점 2자리 까지 출력
print('MSE: {:.2f}'.format(poly_mse))
print('RMSE: {:.2f}'.format(poly_rmse))
print('R2: {:.2f}'.format(poly_r2))
print('score: {:.2f}'.format(poly_score))

#%%
# 학습시킨 모델 저장하기
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import joblib
# %%
#   학습시킨 모델 저장
joblib.dump(model, 'poly.pkl')
# %%
#릿지 회귀
from sklearn.linear_model import Ridge
# %%
# 그리드 서치를 이용한 릿지 회귀 모델 학습
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [0.001, 0.05, 0.1, 1, 10, 20]}
gs_ridge = GridSearchCV(Ridge(), param_grid, cv=5)
gs_ridge.fit(X_train_poly, ytrain)

#%%
# 릿지 회귀 모델 평가MSE, RMSE, R2, score 점수 저장
ridge_mse = mean_squared_error(ytest, gs_ridge.predict(X_test_poly))
ridge_rmse = np.sqrt(mean_squared_error(ytest, gs_ridge.predict(X_test_poly)))
ridge_r2 = r2_score(ytest, gs_ridge.predict(X_test_poly))
ridge_score = gs_ridge.score(X_test_poly, ytest)
ridge_best_params = gs_ridge.best_params_
ridge_best_score = gs_ridge.best_score_
ridge_best_model = gs_ridge.best_estimator_
# 릿지 회귀 모델 평가MSE, RMSE, R2, score 출력
print('릿지 회귀 평가')
print('MSE: ', ridge_mse)
print('RMSE: ', ridge_rmse)
print('R2: ', ridge_r2)
print('score: ', ridge_score)
print('best_params: ', ridge_best_params)
print('best_score: ', ridge_best_score)
# 릿지 계수 출력
print('릿지 계수: ', ridge_best_model.coef_)
# %%
#%%
data = np.array(X_test_poly[0]).reshape(1, -1)
# print()
rbp = ridge_best_model.predict(data)
print(rbp)
print(data)
#%%
#%%
# 학습시킨 모델 저장하기
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import joblib
#%%
#   학습시킨 모델 피클로 저장
joblib.dump(ridge_best_model, 'rid_best.pkl')
#%%
# %%
# 라쏘 회귀
from sklearn.linear_model import Lasso
# %%
# 그리드 서치를 이용한 라쏘 회귀 모델 학습
param_grid = {'alpha': [0.001, 0.05, 0.1, 1, 10, 20]}
gs_lasso = GridSearchCV(Lasso(), param_grid, cv=5)
gs_lasso.fit(X_train_poly, ytrain)
#%%
# 라쏘 회귀 모델 평가MSE, RMSE, R2, score 점수 저장
lasso_mse = mean_squared_error(ytest, gs_lasso.predict(X_test_poly))
lasso_rmse = np.sqrt(mean_squared_error(ytest, gs_lasso.predict(X_test_poly)))
lasso_r2 = r2_score(ytest, gs_lasso.predict(X_test_poly))
lasso_score = gs_lasso.score(X_test_poly, ytest)
lasso_best_params = gs_lasso.best_params_
lasso_best_score = gs_lasso.best_score_
lasso_best_model = gs_lasso.best_estimator_
# 라쏘 회귀 모델 평가MSE, RMSE, R2, score 출력
print('라쏘 회귀 평가')
print('MSE: ', lasso_mse)
print('RMSE: ', lasso_rmse)
print('R2: ', lasso_r2)
print('score: ', lasso_score)
print('best_params: ', lasso_best_params)
print('best_score: ', lasso_best_score)
#%%
#   학습시킨 모델 저장
joblib.dump(lasso_best_model, 'lasso.pkl')
# %%
# 엘라스틱넷 회귀
from sklearn.linear_model import ElasticNet
# %%
# 그리드 서치를 이용한 엘라스틱넷 모델 학습
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'l1_ratio': [0.001, 0.01, 0.1, 1, 10, 100]}
gs_ela = GridSearchCV(ElasticNet(), param_grid, cv=5)
gs_ela.fit(X_train_poly, ytrain)
# %%
# 엘라스틱넷 회귀 모델 평가MSE, RMSE, R2, score 점수 저장
ela_mse = mean_squared_error(ytest, gs_ela.predict(X_test_poly))
ela_rmse = np.sqrt(mean_squared_error(ytest, gs_ela.predict(X_test_poly)))
ela_r2 = r2_score(ytest, gs_ela.predict(X_test_poly))
ela_score = gs_ela.score(X_test_poly, ytest)
ela_best_params = gs_ela.best_params_
ela_best_score = gs_ela.best_score_
ela_best_model = gs_ela.best_estimator_
# 엘라스틱넷 회귀 모델 평가MSE, RMSE, R2, score 출력
print('엘라스틱넷 회귀 평가')
print('MSE: ', ela_mse)
print('RMSE: ', ela_rmse)
print('R2: ', ela_r2)
print('score: ', ela_score)
print('best_params: ', ela_best_params)
print('best_score: ', ela_best_score)
#%%
# 학습시킨 모델 저장하기
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 학습시킨 모델 저장하기
joblib.dump(ela_best_model, 'ela.pkl')

# %%
# 피클로 저장된 엘라스틱넷 회귀 모델 불러오기
#%%
# 랜덤포레스트 회귀
from sklearn.ensemble import RandomForestRegressor
# %%
# 그리드 서치를 이용한 랜덤 포레스트 회귀 모델 학습
param_grid = {'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
gs_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
gs_rf.fit(X_train_poly, ytrain)
#%%
# 랜덤 포레스트 회귀 모델 평가MSE, RMSE, R2, score 점수 저장
rf_mse = mean_squared_error(ytest, gs_rf.predict(X_test_poly))
rf_rmse = np.sqrt(mean_squared_error(ytest, gs_rf.predict(X_test_poly)))
rf_r2 = r2_score(ytest, gs_rf.predict(X_test_poly))
rf_score = gs_rf.score(X_test_poly, ytest)
rf_best_params = gs_rf.best_params_
rf_best_score = gs_rf.best_score_
rf_best_model = gs_rf.best_estimator_
# 랜덤 포레스트 회귀 모델 평가MSE, RMSE, R2, score 출력
print('랜덤 포레스트 회귀 평가')
print('MSE: ', rf_mse)
print('RMSE: ', rf_rmse)
print('R2: ', rf_r2)
print('score: ', rf_score)
print('best_params: ', rf_best_params)
print('best_score: ', rf_best_score)

#%%
# 학습시킨 모델 저장하기
joblib.dump(rf_best_model, 'rf.pkl')
# %%
# 다항회기, 릿지, 라쏘, 엘라스틱넷 회귀 모델 비교
print('다항회귀 평가')
print('MSE: ', poly_mse)
print('RMSE: ', poly_rmse)
print('R2: ', poly_r2)
print('score: ', poly_score)
print('='*50)
print('릿지 회귀 평가')
print('MSE: ', ridge_mse)
print('RMSE: ', ridge_rmse)
print('R2: ', ridge_r2)
print('score: ', ridge_score)
print('best_params: ', ridge_best_params)
print('best_score: ', ridge_best_score)
print('='*50)
#%%
print('라쏘 회귀 평가')
print('MSE: ', lasso_mse)
print('RMSE: ', lasso_rmse)
print('R2: ', lasso_r2)
print('score: ', lasso_score)
print('best_params: ', lasso_best_params)
print('best_score: ', lasso_best_score)
print('='*50)
print('엘라스틱넷 회귀 평가')
print('MSE: ', ela_mse)
print('RMSE: ', ela_rmse)
print('R2: ', ela_r2)
print('score: ', ela_score)
print('best_params: ', ela_best_params)
print('best_score: ', ela_best_score)
print('='*50)
print('랜덤 포레스트 회귀 평가')
print('MSE: ', rf_mse)
print('RMSE: ', rf_rmse)
print('R2: ', rf_r2)
print('score: ', rf_score)
print('best_params: ', rf_best_params)
print('best_score: ', rf_best_score)
print('='*50)
# %%
# 다항회기, 릿지, 라쏘, 엘라스틱넷 회귀 모델 비교 시각화
import matplotlib.pyplot as plt
# %%
# ploy_score, ridge_score, lasso_score, ela_score rf_score 비교
# y축 0.8 ~ 1.0
plt.ylim(0, 1.0)
# x축 레이블
plt.xticks([0, 1, 2, 3, 4], ['poly', 'ridge', 'lasso', 'ela', 'rf'])
# 막대그래프 색상 서로 다르게(파랑색 계열)
plt.bar([0, 1, 2, 3, 4], [poly_score, ridge_score, lasso_score, ela_score, rf_score], color = ['b', 'g', 'r', 'c', 'm'])
plt.show()

# %%
# K_means 군집화
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
#%%
plt.figure(figsize = (10, 6))

for i in range(1, 7):
    # 클러스터 생성
    estimator = KMeans(n_clusters = i)
    # 클러스터링
    estimator.fit(X_test_poly)
    # 클러스터링 결과
    ids = estimator.labels_
    # 2행 3열을 가진 서브플롯 추가 (인덱스 = i)
    plt.subplot(3, 2, i)
    plt.tight_layout()
    # 서브플롯의 라벨링
    plt.title("K value = {}".format(i))
    plt.xlabel('')
    plt.ylabel('')
    # 클러스터링 그리기
    plt.scatter(X_test_poly[:, 0], X_test_poly[:, 1], c=ids, s=50, cmap='viridis')
    plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1], c='black', s=200, alpha=0.5)
plt.show()
# %%
import seaborn as sns

def visualize_inertia(cluster_lists, X_test_poly):
    inertias = []
    for n_cluster in cluster_lists:
        k_means = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        k_means.fit(X_test_poly)
        inertias.append(k_means.inertia_)
        
    sns.lineplot(x=cluster_lists, y=inertias)
visualize_inertia([i for i in range(2, 11)], X_test_poly)
#%%
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def visualize_silhouette(cluster_lists, X_features):     
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)
        
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")

visualize_silhouette([3, 4, 5,6,7,8], X_test_poly)
# %%
# 학습시킨 K-means 모델을 이용해 테스트 데이터를 예측
k_means = KMeans(n_clusters = 6, max_iter=500, random_state=0)
k_means.fit(X_test_poly)
y_pred = k_means.predict(X_test_poly)
print(y_pred)
# %%
# 예측 결과를 시각화
plt.scatter(X_test_poly[:, 0], X_test_poly[:, 1], c=y_pred, s=50, cmap='viridis')
centers = k_means.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
# %%
# K-means 학습모델 저장
joblib.dump(k_means, 'k_means.pkl')
# %%
# 값을 입력받아서 다항회기 모델을 이용해 예측한 결과를 시각화하는 함수


def poly_plt(data):
    # 입력받은 데이터를 다항회기 모델을 이용해 예측
    y_pred = model.predict(data)
    # 예측 결과를 시각화
    plt.scatter(X_test_poly[:, 0], X_test_poly[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.scatter(data[:, 0], data[:, 1], c='black', s=200, alpha=0.5)
    plt.show()        
    
poly_plt(X_test_poly[0].reshape(1, -1))
# %%
