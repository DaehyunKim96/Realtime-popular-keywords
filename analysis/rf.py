from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
import codecs
import time
import numpy as np
import pandas as pd

def cal_crawltime(cr_time, date):
    cr_time = int(cr_time.split('/')[1])*60
    date = date.split(' ')
    check, time = date[1:3]
    dt = -1
    if check == '오전':
        dt = int(time.split(':')[0]) * 60 + int(time.split(':')[1])
    else:
        dt = (int(time.split(':')[0])+12) * 60 + int(time.split(':')[1])
    return cr_time - dt

def comma_elim(val:str):
    if val == '':
        return 0
    else:
        if ',' in val:
            return int(''.join(val.split(',')))
        else:
            return int(val)

def naver_rf(json_data:list):
    X,y = [],[]
    for elem in json_data:
        elem = elem['nc']['properties']
        tmp = []
        #tmp.append(cal_crawltime(elem['crawl_time'], elem['date']))
        try:
            rg, rw, rs, ra, rwt = comma_elim(elem['good']), comma_elim(elem['warm']), comma_elim(elem['sad']), comma_elim(elem['angry']), comma_elim(elem['want'])
            features = [rg+rw+rs+ra+rw, rg, rw, rs, ra, rwt, elem['n_comment'],elem['pagerank'],elem['betweeness'],elem['closeness'],elem['n_view'],elem['degree']]
            for f in features:
                tmp.append(f)
            y.append(elem['rank'])
            X.append(tmp)
        except:
            pass
    feature = ['reactions','good','warm','sad','angry','want','n_comment','pagerank','betweeness','closeness','n_view','degree']
    return X,y,feature

def daum_rf(json_data:list):
    X,y = [],[]
    for elem in json_data:
        elem = elem['nc']['properties']
        tmp = []
        try:
            #tmp.append(cal_crawltime(elem['crawl_time'], elem['date']))
            rl, ri, ra, rs, rr = elem['like'], elem['impress'], elem['angry'], elem['sad'], elem['recommend']
            features = [rl+ri+ra+rs+rr, rl, ri, ra, rs, rr, elem['n_comment'],elem['pagerank'],elem['betweeness'],elem['closeness'],elem['degree']]
            for f in features:
                tmp.append(f)
            y.append(elem['rank'])
            X.append(tmp)
        except:
            pass
    feature = ['reactions','like','impress','angry','sad','recommend','n_comment','pagerank','betweeness','closeness','degree']
    return X,y,feature

def youtube_rf(json_data:list):
    X,y = [],[]
    for elem in json_data:
        elem = elem['nc']['properties']
        tmp = []
        try:
            #tmp.append(cal_crawltime(elem['crawl_time'], elem['date']))
            rg, rb = elem['good'], elem['bad']
            features = [rg+rb, rg, rb, elem['n_comment'],elem['n_view'],elem['pagerank'],elem['betweeness'],elem['closeness'],elem['degree']]
            for f in features:
                tmp.append(f)
            y.append(elem['rank'])
            X.append(tmp)
        except:
            pass
    feature = ['reactions','good','bad','n_comment','n_view','pagerank','betweeness','closeness','degree']
    return X,y,feature

def linear(X:list, y:list,feature:list):
    model = LinearRegression()
    model.fit(X, y)
    importance = model.coef_
    for i,v in enumerate(importance):
        print(str(feature[i])+', Score: %.5f' % (v))
        
def logistic(X:list, y:list,feature:list):
    model = LogisticRegression()
    model.fit(X, y)
    importance = model.coef_[0]
    for i,v in enumerate(importance):
        print(str(feature[i])+', Score: %.5f' % (v))

def cart(X:list, y:list,feature:list):
    model = DecisionTreeRegressor()
    model.fit(X, y)
    importance = model.feature_importances_
    for i,v in enumerate(importance):
        print(str(feature[i])+', Score: %.5f' % (v))

def rf(X:list, y:list,feature:list):
    #X_data = np.array(X, dtype = int)
    #y_data = np.array(y, dtype = int)
    #X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X, y)

    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: "
        f"{elapsed_time:.3f} seconds")
    
    forest_importances = pd.Series(importances, index=feature)
    print(forest_importances)

def xgboost(X:list, y:list,feature:list):
    model = XGBClassifier()
    model.fit(X, y)
    importance = model.coef_
    for i,v in enumerate(importance):
        print(str(feature[i])+', Score: %.5f' % (v))
        
if __name__ == "__main__":
    with codecs.open("/home/capje/analysis/0522_youtube.json",'r','utf-8-sig') as f:
        json_data = json.load(f)
    X,y,feature = youtube_rf(json_data)
    #범주형 데이터는 부적합
    #linear(X,y,feature)
    #logistic은 바이너리에 적합
    #logistic(X,y,feature)

    rf(X,y,feature)

