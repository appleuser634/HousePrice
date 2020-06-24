import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn import tree 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import threading 
import argparse
import sys


#データーセットの読み込み
def load_dataset(path):
    
    df = pd.read_csv(path)
    
    return df


#訓練用データーからテストデーターを分ける
def split_dataset(df,train_per):

    data_n = len(df)
    train_per = int(data_n * (train_per/100))
    
    train_data = df[:train_per]
    test_data = df[train_per:]
    
    print("Train Datas:",len(train_data))
    print("Test Datas:",len(test_data))

    return train_data,test_data


#データーセット情報を提示する関数
def show_datainfo(df):
 
    #データーの表示
    print("\nTrain Data Content:")
    print(df.head())
    
    print(df.info())


#どの変数を仕様するか決定する関数
def chose_val(df,response_variable_name,values_n):
    
    df_coor = df.corr()

    #学習データーの各変数の相関係数を表示
    print("Data Core:")
    print(df_coor)
    
    df_coor = df_coor.drop(response_variable_name, axis=0)
    
    Important_datas = []
    Important_value = []

    for _ in range(values_n):
        
        important_data = df_coor[response_variable_name].idxmax()
        Important_datas.append(important_data)
        
        important_value = df_coor.at[important_data,response_variable_name]
        Important_value.append(important_value)

        df_coor = df_coor.drop(important_data, axis=0)
        
        important_data = df_coor[response_variable_name].idxmin()
        Important_datas.append(important_data)

        important_value = df_coor.at[important_data,response_variable_name]
        Important_value.append(important_value)

        df_coor = df_coor.drop(important_data, axis=0)
    

    print("\nImportant Datas:") 
    for i,value_name in enumerate(Important_datas):
        print(value_name,Important_value[i])

    Important_datas.append(response_variable_name)
    Important_dataframe = df[[d for d in Important_datas]]
    
    print("\nImportant_dataframe head:\n",Important_dataframe.head())

    return Important_dataframe


#データセットに対し前処理（ラベルエンコード欠損の補完など）
def data_processing(df):

    for i in range(df.shape[1]):
        if df.iloc[:,i].dtypes == object:
            lbl = LabelEncoder()
            lbl.fit(list(df.iloc[:,i].values) + list(df.iloc[:,i].values))
            df.iloc[:,i] = lbl.transform(list(df.iloc[:,i].values)) 

    print("Processed Dataframe:",df.head())    
    return df

#テストデーターを用いて予測スコアを計算
def cal_score(train_pred_y,train_y,test_pred_y,test_y):
    
    print('\nTrain Data Score: ', r2_score(train_y, train_pred_y))
    print('Test Data Score: ', r2_score(test_y, test_pred_y))

#並列にグリッドサーチを実行
def thread1(train_X,train_y):
    
    #グリッドサーチを行うランダムフォレストを宣言
    param_range = [10,20,30]
    paramG = {'max_depth':param_range,'n_estimators':param_range,'random_state':param_range}
    RFC_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=paramG, scoring='r2', cv=3)
    RFC_grid.fit(train_X,train_y.ravel())
    
    print("thread1 Best param:")
    print("max_depth:",RFC_grid.best_estimator_.max_depth)
    print("n_estimators:",RFC_grid.best_estimator_.n_estimators)
    print("random_state:",RFC_grid.best_estimator_.random_state)

#並列にグリッドサーチを実行
def thread2(train_X,train_y):
    
    #グリッドサーチを行うランダムフォレストを宣言
    param_range = [40,50,60]
    paramG = {'max_depth':param_range,'n_estimators':param_range,'random_state':param_range}
    RFC_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=paramG, scoring='r2', cv=3)
    RFC_grid.fit(train_X,train_y.ravel())
    
    print("thread1 Best param:")
    print("max_depth:",RFC_grid.best_estimator_.max_depth)
    print("n_estimators:",RFC_grid.best_estimator_.n_estimators)
    print("random_state:",RFC_grid.best_estimator_.random_state)

#並列にグリッドサーチを実行
def thread3(train_X,train_y):
    
    #グリッドサーチを行うランダムフォレストを宣言
    param_range = [70,80,90]
    paramG = {'max_depth':param_range,'n_estimators':param_range,'random_state':param_range}
    RFC_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=paramG, scoring='r2', cv=3)
    RFC_grid.fit(train_X,train_y.ravel())
    
    print("thread1 Best param:")
    print("max_depth:",RFC_grid.best_estimator_.max_depth)
    print("n_estimators:",RFC_grid.best_estimator_.n_estimators)
    print("random_state:",RFC_grid.best_estimator_.random_state)

#処理フローをまとめるメイン関数
def main(path,response_variable_name,values_n):
    
    #データーセットの読み込み
    df = load_dataset(path)
    #データーセットの内容を表示する
    show_datainfo(df)
    #ラベルエンコードや前処理を行う    
    df = data_processing(df)    
    #相関の高い変数だけを抽出する
    Important_dataframe = chose_val(df,response_variable_name,values_n)
 
    #訓練用データーからテストデータを分ける
    train_data,test_data = split_dataset(Important_dataframe,75)
    
    #訓練用データーから目的変数を取り除く
    train_X = train_data.drop(response_variable_name, axis=1)
    #訓練用データーの目的変数を宣言する
    train_y = train_data[[response_variable_name]].values

    #テストデーターから目的変数を取り除く
    test_X = test_data.drop(response_variable_name, axis=1)
    #テストデーターの目的変数を宣言する
    test_y = test_data[[response_variable_name]].values

    #線形回帰モデルを宣言
    linear = LinearRegression()    
    #モデル作成
    linear.fit(train_X,train_y)

    #決定木モデルを宣言
    clf = tree.DecisionTreeClassifier(max_depth=values_n*2)
    #モデル作成
    clf.fit(train_X,train_y)
    
    #ランダムフォレストモデルを宣言
    random_forest = RandomForestClassifier(max_depth=30, n_estimators=91, random_state=42)
    random_forest.fit(train_X,train_y.ravel())
    
    #グリッドサーチを行うランダムフォレストを宣言　並列化して3スレッドで実行
    grid_thread1 = threading.Thread(target=thread1,args=([train_X,train_y]))
    grid_thread2 = threading.Thread(target=thread2, args=([train_X,train_y]))
    grid_thread3 = threading.Thread(target=thread3, args=([train_X,train_y]))
    
    grid_thread1.start()
    grid_thread2.start()
    grid_thread3.start()

    #paramG = {'max_depth':[i for i in range(1,100,10)],'n_estimators':[i for i in range(1,100,10)],'random_state':[i for i in range(1,100,10)]}
    #RFC_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=paramG, scoring='r2', cv=3)
    #RFC_grid.fit(train_X,train_y.ravel())

    #print("Best param:")
    #print("max_depth:",RFC_grid.best_estimator_.max_depth)
    #print("n_estimators:",RFC_grid.best_estimator_.n_estimators)
    #print("random_state:",RFC_grid.best_estimator_.random_state)

    #XGBoostモデルを宣言
    xgb = XGBClassifier()
    xgb.fit(train_X,train_y)

    #k近傍モデルを宣言
    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(train_X, train_y)

    #勾配ブースティングモデルを宣言
    #gbrt = GradientBoostingClassifier(random_state=0)
    #gbrt.fit(train_X,train_y)

    #線形回帰による予測
    print("\n線形回帰によるスコア:")
    train_pred_y = linear.predict(train_X)
    test_pred_y = linear.predict(test_X)
    cal_score(train_pred_y,train_y,test_pred_y,test_y)    
    
    #決定木による予測
    print("\n決定木によるスコア:")
    tree_train_predicted_y = clf.predict(train_X)
    tree_test_predicted_y = clf.predict(test_X)
    cal_score(tree_train_predicted_y,train_y,tree_test_predicted_y,test_y)
    
    #ランダムフォレストによる予測
    print("\nランダムフォレストによるスコア:")
    rf_train_predicted_y = random_forest.predict(train_X)
    rf_test_predicted_y = random_forest.predict(test_X)
    cal_score(rf_train_predicted_y,train_y,rf_test_predicted_y,test_y)

    #ランダムフォレストによる予測
    print("\nランダムフォレスト(グリッドサーチ)によるスコア:")
    rfg_train_predicted_y = RFC_grid.predict(train_X)
    rfg_test_predicted_y = RFC_grid.predict(test_X)
    cal_score(rf_train_predicted_y,train_y,rf_test_predicted_y,test_y)
    
    #XGBoostによる予測
    print("\nXGBoostによるスコア:")
    xgb_train_predicted_y = xgb.predict(train_X)
    xgb_test_predicted_y = xgb.predict(test_X)
    cal_score(xgb_train_predicted_y,train_y,xgb_test_predicted_y,test_y)

    #K近傍による予測
    print("\nK近傍によるスコア:")
    knc_train_predicted_y = knc.predict(train_X)
    knc_test_predicted_y = knc.predict(test_X)
    cal_score(knc_train_predicted_y,train_y,knc_test_predicted_y,test_y)

    #勾配ブースティングによる予測
    #print("\n勾配ブースティングによるスコア:")
    #train_predicted_y = clf.predict(train_X)
    #test_predicted_y = clf.predict(test_X)
    #cal_score(train_predicted_y,train_y,test_predicted_y,test_y)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='ML Programm.')

    data_path = sys.argv[1]    
    response_variable_name = sys.argv[2]
    values_n = int(sys.argv[3])

    main(data_path,response_variable_name,values_n)
