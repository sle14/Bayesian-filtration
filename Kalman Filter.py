import matplotlib.pyplot as plt
from scipy import stats as st
import numpy as np

np.random.seed(seed=123)

def LLF(R, y, y_hat):
    f = st.norm(0, R).pdf(y - y_hat)
    return -np.sum(np.log(f))

def cons(R): #force cov to be positive
    return R[-1]

def censor(p, T):
    lst = [0, *np.random.choice([1,0],size=T-1,p=[p,1-p]).tolist()]
    return np.nonzero(np.array(lst))[0] 

def reg(y, lags=1, trend="c"):
    from statsmodels.tsa.ar_model import AutoReg
    ar = AutoReg(y, lags=lags, trend=trend).fit()
    print("\n",ar.params,
          "\n",np.sqrt(np.diag(ar.cov_params())),
          "\n",np.var(ar.resid,ddof=1))
    
def ar(a,b,T):
    y = np.zeros(T)
    y[0] = 0
    for t in range(1,T): 
        y[t] = a + b*y[t-1] + st.norm.rvs(0, 1)
    return y


#-------------------------------------------------------------------------
    
T = 200
a = 0
b = 0.5

y = ar(a,b,T) 
# plt.plot(ys)

beta = np.array([a,b])
Y = np.array([np.ones(T),y]).T

# Y[censor(0.1,T),:] = np.nan

#-------------------------------------------------------------------------

nx = len(beta)
ny = 1

#State noise cov
Q = 0.001 * np.eye(nx)

#Measurement noise cov
R = 1 * np.eye(ny) #note how incorrect R reduces P estimates accuracy

#Filtering pass outputs
m = np.zeros((T,nx))
P = np.zeros((T,nx,nx))
m[0] = 0
P[0] = 2 * np.eye(nx)

#Smoothing pass outputs
L = T // 10
ms = np.zeros((T,nx))
Ps = np.zeros((T,nx,nx))
ms[0] = 0
Ps[0] = 2 * np.eye(nx)

#-------------------------------------------------------------------------

est_cov = False #estimate measurement noise covariance
const_x = False #perform recursive regression

for t in range(T-1):
    F = np.eye(nx)
    H = np.array([Y[t]])
 
    #-------------------------------------------------------------------------
    # Filtering pass
    #-------------------------------------------------------------------------
    
    # PREDICTION - from p(x[t-1]|y[t-1]) to p(x[t]|y[t-1])
 
    #Constant state = recursive regression = converge to batch regression
    if const_x == True:
        x_hat = m[t]
        P_hat = P[t]
    else:
        x_hat = F @ m[t]
        P_hat = (F @ P[t] @ F.T) + Q 
        
    # UPDATE - from p(x[t]|y[t-1]) to p(x[t]|y[t])
    
    #Handle missing curr observations by propogating with extra variance
    if np.isnan(Y[t,:]).all() == True:
        m[t+1] = x_hat
        P[t+1] = P_hat
        continue   
    
    #Estimate observation noise covariance empirically
    if est_cov == True and t > 1:
        eps = Y[1:t] - Y[0:t-1] @ np.array([m[t]]).T
        eps = eps[~np.isnan(eps)]
        R = np.var(eps) if len(eps) > 0 else R
    
    S = (H @ P_hat @ H.T) + R
    K = (P_hat @ H.T) @ np.linalg.inv(S) 
    y_hat = H @ x_hat
    eps = (y[t+1] - y_hat)
    
    #Handle missing next observation by zeroing innovation
    if np.isnan(Y[t+1,:]).all() == True:
        eps = np.array([0])
    
    m[t+1] = x_hat + K @ eps
    P[t+1] = P_hat - K @ H @ P_hat

    #-------------------------------------------------------------------------
    # Smoothing pass
    #-------------------------------------------------------------------------
    
    ms[t+1] = m[t+1]
    Ps[t+1] = P[t+1]
    t_L = max(t-L,0)
    for s in range(t,t_L-1,-1):
        F = np.eye(nx)
        
        #Constant state = recursive regression = converge to batch regression
        if const_x == True:
            x_hat = m[s]
            P_hat = P[s]
        else:
            x_hat = F @ m[s]
            P_hat = (F @ P[s] @ F.T) + Q 
        
        K = P[s] @ F.T @ np.linalg.inv(P_hat)   
        ms[s] = m[s] + K @ (ms[s+1] - x_hat)
        Ps[s] = P[s] + K @ (Ps[s+1] - P_hat) @ K.T


#-------------------------------------------------------------------------
P_ = np.diagonal(P,axis1=1,axis2=2)
for n in range(nx):
    plt.axhline(beta[n], c="black", linewidth = 0.8)
    plt.plot(m[:,n], label=f"m{n}", alpha=0.8)
    plt.fill_between(range(T),
                      y1 = m[:,n] - 1.96*np.sqrt(P_[:,n]),
                      y2 = m[:,n] + 1.96*np.sqrt(P_[:,n]), 
                      alpha=0.2)
    plt.plot(ms[:,n], label=f"ms{n}", alpha=0.8)
    if np.isnan(Y).any() == True:
        plt.scatter(range(T),np.where(np.isnan(Y),np.nan,-3)[:,0],
                    s=12,c="black",alpha=0.1)
    plt.legend()
    plt.show()

#-------------------------------------------------------------------------

reg(y)
print("\n",m[-1], "\n",np.sqrt(P_[-1]), "\n",R)