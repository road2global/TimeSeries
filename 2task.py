import numpy as np

def Theta(z):
        if z < 0:
            return 0
        return 1
def p(k, n1, n2, y):
	w_n1 = y[n1-k:n1]
	w_n2 = y[n2-k:n2]
	S = np.array([(w_n1[i]-w_n2[i])**2 for i in range(k)])
	return np.sqrt(np.sum(S))
a = 3
dt = 0.001
N = 10000
Time_Series = [i * dt for i in range(N)]
y = np.array([a * np.sin(t) for t in Time_Series])
l = 0.001
for k in range(2, 15):
    n0 = N - k
    L = np.array([[Theta(l-p(k, i, j, y)) for j in range(n0, N)] for i in range(n0, N)])
    C = (1 / N**2) * np.sum(L)
    d = np.log(C)/np.log(l)
    print(d)
