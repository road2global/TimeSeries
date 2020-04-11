import numpy as np
import requests
from json import loads
from time import time
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime as datetime

def la(data, p, second_order=True, eps=1e-5):
    Xi = 3 * (p + 1)
    X = data[np.arange(p) + np.arange(len(data) - p)[:, None]]
    omega = np.argpartition(np.sum(np.power(X - data[-p:], 2), axis=1), Xi)[:Xi]
    if not second_order:
        Y = np.hstack((np.ones(Xi)[:, None], X[omega]))
    else:
        idx = np.arange(p)[:, None] - np.arange(p) <= 0
        Y = np.hstack((np.ones(Xi)[:, None], (X[omega, :, None] * X[omega, None, :])[:, idx]))
    params = np.linalg.solve(Y.T @ Y + eps * np.eye(Y.shape[1]), Y.T @ data[omega + p])
    if not second_order:
        return params, np.sum(params * np.hstack([1, data[-p:]]))
    else:
        return params, np.sum(params * np.hstack([1, (data[-p:, None] * data[-p:])[idx]]))

def one_iter_predict(method, n, X):
    ans = np.array([])
    X_L = X
    for _ in range(n):
        pred = method(X_L, p)[1]
        ans = np.append(ans, pred)
        X_L = np.append(X_L, pred)
    return ans
start_time = time() - 168*60*60
resource = requests.get("https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=%s&end=9999999999&period=1800" % start_time)
data = loads(resource.text)

quotes = {}
quotes['open']=np.asarray([item['open'] for item in data])
quotes['close']=np.asarray([item['close'] for item in data])
quotes['high']=np.asarray([item['high'] for item in data])
quotes['low']=np.asarray([item['low'] for item in data])

sign = lambda x : 1 if x >= 0 else -1
X = quotes['close'] - quotes['open']
y = [sign(elem) for elem in X]
df = DataFrame(list(zip(X, y)))
df.columns = ['close-open', 'class']
data = df['close-open'].values
N = len(data)

plt.plot(np.arange(N), data, 'b')
plt.ylabel('close-open $')
plt.xlabel('Days')
plt.show()

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

scores = {}
for k in range(2, 20):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    scores[k] = accuracy_score(y_test, y_pred)

p = max(scores, key=lambda k: scores[k])

la_predict = one_iter_predict(la, 30, data)
la_predict[la_predict > 0], la_predict[la_predict < 0] = 1, -1

plt.plot(np.arange(len(la_predict)), la_predict, color='b')
plt.xlabel('Days')
plt.ylabel('Predicted')
plt.show()
