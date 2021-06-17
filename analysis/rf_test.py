from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
import codecs
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

with codecs.open("/home/capje/analysis/test_degree.json",'r','utf-8-sig') as f:
    json_data = json.load(f)

X = []
y = []

for elem in json_data:
    elem = elem['n']['properties']
    tmp = []
    #tmp.append(cal_crawltime(elem['crawl_time'], elem['date']))
    rg, rw, rs, ra, rwt = comma_elim(elem['good']), comma_elim(elem['warm']), comma_elim(elem['sad']), comma_elim(elem['angry']), comma_elim(elem['want'])
    features = [rg+rw+rs+ra+rw, rg, rw, rs, ra, rwt, elem['n_comment'],elem['pagerank'],elem['betweeness'],elem['closeness'],elem['n_view'],elem['degree']]
    for f in features:
        tmp.append(f)
    y.append(elem['rank'])
    X.append(tmp)

X_data = np.array(X, dtype = int)
print(len(X_data[0]))
X_data_2 = np.transpose(X_data)
print(len(X_data_2[0]))