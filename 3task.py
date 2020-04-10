from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ext_series(s):
    ans = 0
    if (1 <= s <= tau):
        for i in range(1, s+1):
            ans += new_X[i-1][s-i]
        return ans / s
    elif (tau <= s <= n):
        for i in range(1, tau+1):
            ans += new_X[i-1][s-i]
        return ans / (tau+1)
    elif (n <= s <= N):
        for i in range(1, N-s+2):
            ans += new_X[i+s-n-2][n-i]
        return ans / (N-s+1)

def Draw(series, color, xlabel, ylabel):
    plt.plot(series, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
data = pd.read_csv('toyota.csv', sep =';')
data = data.drop(['<TIME>', '<HIGH>','<LOW>','<CLOSE>', '<VOL>'], axis=1)
data['<DATE>'] = pd.to_datetime(data['<DATE>'], format="%Y%m%d")
print(type(data['<OPEN>'].values))
date_series = data['<DATE>'].values
pr_series = data['<OPEN>'].values
N = len(pr_series)
tau = (N + 1) // 2
n = N + 1 - tau
X = np.array([pr_series[i: i + n] for i in range(tau)])
#print(X)

Lambda, V = np.linalg.eig(X @ X.T / n)

Y = V.T.dot(X)
r = 15
V_r = V[:,:r]
Y_r = V.T.dot(X)[:r,]
new_X = V_r.dot(Y_r)
smooth_pr_series = [ext_series(i) for i in range(1, N+1)]

plt.plot(pr_series, color='r')
plt.plot(smooth_pr_series, color ='b')
plt.xlabel('Price')
plt.ylabel('Days')
plt.legend(["Series", "Smooth_Series"])
plt.show()

V_tau = V[-1, :r]
V_ast = V[:tau-1, :r]
Q = X[-tau+1 :]

denom = V_tau @ V_ast.T
delim = 1 - V_tau @ V_tau.T
predict_series = (denom.dot(Q)) / delim

predict_series = np.append(pr_series[:N-n], predict_series)

plt.plot(pr_series, 'b')
plt.plot(predict_series, 'r')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend(["Series","Predict_Series"])
plt.show()
