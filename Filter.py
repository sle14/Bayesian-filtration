from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from scipy import stats as st
import numpy as np

np.random.seed(seed=123)

class Filter:
    def __init__(self,y,T,F,H,Q,R,m0,P0):
        '''
        Params
        --------
        y : observations 
        F : transition model mapping x[t-1] to x[t]
        H : observation model mapping x[t] to y[t]
        Q : state noise covariance
        R : measurement noise covariance
        m0 : prior mean
        P0 : prior covariance
        '''

        self.y = y 
        self.T = T
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.R_ = R
    
        nx = len(np.diag(Q))
        self.nx = nx
        
        self.m = np.zeros((T,nx))
        self.P = np.zeros((T,nx,nx))
        self.m[0] = m0
        self.P[0] = P0
        
    
    def Kalman(self, est_cov=False, const_x=False):
        '''
        Params
        --------
        est_cov : estimate measurement noise covariance
        const_x : perform recursive regression
        '''
        
        y = self.y 
        F = self.F
        H = self.H
        Q = self.Q
        R = self.R_ if est_cov==False else 1e-3
        m = self.m
        P = self.P
        T = self.T        
        
        for t in range(T-1):
            H_t = np.array([H[t]]) 
                
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
            
            #No observation - continue to diffuse
            if np.isnan(y[t]) == True:
                m[t+1] = x_hat
                P[t+1] = P_hat
                continue   
            
            #Estimate observation noise covariance empirically
            if est_cov == True and t > 1:
                eps = np.array([y[:t]]).T - H[:t] @ np.array([m[t]]).T
                eps = eps[~np.isnan(eps)]
                R = np.var(eps) if len(eps) > 0 else R
            
            S = (H_t @ P_hat @ H_t.T) + R
            K = (P_hat @ H_t.T) @ np.linalg.inv(S) 
            y_hat = H_t @ x_hat
            eps = y[t] - y_hat
            
            m[t+1] = x_hat + K @ eps
            P[t+1] = P_hat - K @ H_t @ P_hat
        
            #-------------------------------------------------------------------------
            # Smoothing pass
            #-------------------------------------------------------------------------
            
            # ms[t+1] = m[t+1]
            # Ps[t+1] = P[t+1]
            # t_L = max(t-L,0)
            # for s in range(t,t_L-1,-1):
                
            #     #Constant state = recursive regression = converge to batch regression
            #     if const_x == True:
            #         x_hat = m[s]
            #         P_hat = P[s]
            #     else:
            #         x_hat = F @ m[s]
            #         P_hat = (F @ P[s] @ F.T) + Q 
                
            #     K = P[s] @ F.T @ np.linalg.inv(P_hat)   
            #     ms[s] = m[s] + K @ (ms[s+1] - x_hat)
            #     Ps[s] = P[s] + K @ (Ps[s+1] - P_hat) @ K.T
                    
        self.m = m
        self.P = P
        self.R = R
        
    def Particle(self, oid=False, N=1000, c=0.1):
        '''
        N : number of particles to draw
        c : resampling coefficient
        oid : use optimal importance density under gaussian conjugacy and additive noise
        '''

        y = self.y 
        F = self.F
        H = self.H
        Q = self.Q
        R = self.R
        m = self.m
        P = self.P
        T = self.T
        nx = self.nx
        
        #Careful: multivariate expects var arg not std arg as univariate one!
        x = m[0] + st.multivariate_normal(np.zeros(nx), P[0]).rvs(N)
        w = 1 / N 
        
        for t in range(T-1):
            H_t = np.array([H[t]])
            
            # PREDICTION - from p(x[t-1]|y[t-1]) to p(x[t]|y[t-1])
                    
            #Draw particles
            if oid == False or np.isnan(y[t]) == True:
                x_hat = (F @ x.T).T + st.multivariate_normal(np.zeros(nx), Q).rvs(N)
            else:
                S = (H_t @ Q @ H_t.T) + R
                eps = y[t] - (H_t @ F @ x.T).T
                μ = (F @ x.T + Q @ H_t.T @ np.linalg.inv(S) @ eps.T).T
                Σ = Q - Q @ H_t.T @ np.linalg.inv(S) @ H_t @ Q
                x_hat = μ + st.multivariate_normal(np.zeros(nx), Σ).rvs(N)
            
            # UPDATE - from p(x[t]|y[t-1]) to p(x[t]|y[t])
         
            #No observation - continue to diffuse
            if np.isnan(y[t]) == True:
                x = x_hat
                m[t+1] = sum(w * x)
                P[t+1] = sum(w * (x - m[t+1])**2) 
                continue   
            
            #Forecast observation given particle
            y_hat = (H_t @ x_hat.T).T
        
            #Get likelihoods of observation given particles (Particle gain)
            if oid == False:
                a = st.norm(y[t], np.sqrt(R)).pdf(y_hat)
            else:
                a = st.norm(y[t], np.sqrt(S)).pdf(y_hat)
            
            #Update weights with likelihoods
            w = w * a
            
            #Normalise weights
            w = w / sum(w)
        
            #Resample - pass particles through survival function
            N_eff = 1 / sum(w**2)
            if N_eff < c*N: 
                ξ = np.random.choice(range(N), size=N, p=w.flatten())
                x_hat = x_hat[ξ]
                w = 1 / N
        
            #Get posterior moments
            x = x_hat
            m[t+1] = sum(w * x)
            P[t+1] = sum(w * (x - m[t+1])**2) 
    
        self.m = m
        self.P = P

    def GP(self, J, Ψ, N=1000, c=1):
 
        y = self.y 
        F = self.F
        Q = self.Q
        R = self.R
        m = self.m
        P = self.P
        T = self.T
        nx = self.nx
        
        def h(x,j): #observation model
            if j == 1:
                return x[:,1] - Ψ*np.exp(x[:,0])
            else:
                return x[:,1] + Ψ*np.exp(x[:,0]) 
        
        x = m[0] + st.multivariate_normal(np.zeros(nx), P[0]).rvs(N)
        w = 1 / N
        
        for t in range(T-1):
          
            # PREDICTION - from p(x[t-1]|y[t-1]) to p(x[t]|y[t-1])
          
            #Draw particles from prior
            x_hat = (F @ x.T).T + st.multivariate_normal(np.zeros(2), Q).rvs(N)
        
            # UPDATE - from p(x[t]|y[t-1]) to p(x[t]|y[t])
            
            #No observation - continue to diffuse
            if J[t] == 0:
                x = x_hat
                m[t+1] = sum(w * x)
                P[t+1] = sum(w * (x - m[t+1])**2) 
                continue
            
            #Forecast observation given particle
            y_hat = h(x,J[t])
             
            #Obtain weights from likelihood
            a = st.norm(y[t], np.sqrt(R)).pdf(y_hat) #var is odditive
            w = w * a 
            w = np.where(w==.0, 1e-6, w) #handle overflow
            w = w / sum(w)
        
            #Resample - pass particles through survival function
            N_eff = 1 / sum(w**2)
            if N_eff < c*N: 
                ξ = np.random.choice(range(N), size=N, p=w.flatten())
                x_hat = x_hat[ξ]
                w = 1 / N
        
            #Get posterior moments
            x = x_hat
            m[t+1] = sum(w * x)
            P[t+1] = sum(w * (x - m[t+1])**2) 

        self.m = m
        self.P = P
        
    def plot(self, coef:list):
        P_ = np.diagonal(self.P,axis1=1,axis2=2)
        for n in range(self.nx):
            plt.axhline(coef[n], c="black", linewidth = 0.8, label=f"x{n}")
            plt.plot(self.m[:,n], label=f"m{n}", alpha=0.8)
            plt.fill_between(range(self.T),
                              y1 = self.m[:,n] - 1.96*np.sqrt(P_[:,n]),
                              y2 = self.m[:,n] + 1.96*np.sqrt(P_[:,n]), 
                              alpha=0.2)
            plt.legend()
            plt.show()
            
    def plot_GP(self, x_):
        P_ = np.diagonal(self.P,axis1=1,axis2=2)
        plt.plot(x_[:,1], label="Hidden Mid-YtB", c="black", linewidth = 0.8)
        plt.plot(self.m[:,1], label="Estimated Mid-YtB", alpha=0.8)
        plt.fill_between(range(self.T),
                          y1 = self.m[:,1] - 1.96*np.sqrt(P_[:,1]),
                          y2 = self.m[:,1] + 1.96*np.sqrt(P_[:,1]), 
                          alpha=0.2)
        plt.scatter(range(self.T), self.y, label="Quote", s=3, c="red")
        plt.legend()
        plt.show()
        

class OLS:
    def __init__(self,y):
        self.y = y
        self.coef = None #Estimated coefficients
        self.stderr = None #Standard error of estimates
        self.resvar = None #Residual variance
        
    def reg(self, lags=1, trend="c"):
        ar = AutoReg(self.y, lags=lags, trend=trend).fit()
        print(ar.summary())
        self.coef = ar.params 
        self.stderr = ar.cov_params()
        self.resvar = np.var(ar.resid,ddof=1)
                   
def censor(y, H, p, T):
    y,H = np.copy(y),np.copy(H)
    lst = [0, *np.random.choice([0,1],size=T-1,p=[p,1-p]).tolist()]
    idx = np.nonzero(np.array(lst))[0]
    y[idx] = np.nan
    H[idx] = np.nan
    return y,H
    
def ar(a,b,sig2,T):
    y = np.zeros(T)
    y[0] = a + st.norm.rvs(0, 1)
    for t in range(1,T): 
        y[t] = a + b*y[t-1] + st.norm.rvs(0, np.sqrt(sig2))
    return y

def gp(Ψ,A,V,σ,ς,T,dt,PQ):
    ψ = np.zeros(T)
    y = np.zeros(T)
    x = np.zeros(T)
    Q = np.zeros(T); Q[:] = np.nan
    J = np.zeros(T)
    
    y0 = 0.1
    x0 = 0.01

    y[0] = y0
    x[0] = x0
    ψ[0] = Ψ*np.exp(x0)
    
    dW = st.norm(0,np.sqrt(dt)).rvs(T)
    dB = st.norm(0,np.sqrt(dt)).rvs(T)
    
    if np.random.random() < .5:
        Q[0] = y[0] - ψ[0] #Observe offer quote
        J[0] = 1
    else:
        Q[0] = y[0] + ψ[0] #Observe bid quote
        J[0] = 2
                
    for t in range(T-1):
        #Bond YtB process
        y[t+1] = y[t] + σ*dW[t]
    
        #Untransformed spread process
        x[t+1] = x[t] + -A*x[t]*dt + V*dB[t]
    
        #Spread process
        ψ[t+1] = Ψ*np.exp(x[t+1])
        
        if np.random.random() < PQ:
            if np.random.random() < .5:
                Q[t+1] = y[t+1] - ψ[t+1] #Observe offer quote
                J[t+1] = 1
            else:
                Q[t+1] = y[t+1] + ψ[t+1] #Observe bid quote
                J[t+1] = 2
                
    return [x,y,Q,J]

#-------------------------------------------------------------------------
#AR PROCESS
#-------------------------------------------------------------------------
    
# T = 200
# a = 0.1
# b = 0.5
# sig2 = 1

# y_ = ar(a,b,sig2,T) 
# # plt.plot(ys);plt.show()

# #-------------------------------------------------------------------------

# nx = 2
# ny = 1

# #State noise cov
# Q = 0.001 * np.eye(nx)

# #Measurement noise cov
# R = 1 * np.eye(ny) #note how incorrect R reduces P estimates accuracy

# #Transition model
# F = np.eye(nx)

# #Observation model
# H = np.array([np.ones(T-1),y_[:-1]]).T

# #Observations
# y = y_[1:]

# #Initial estimates for mean and cov
# m0 = 0
# P0 = 1 * np.eye(nx)

# #Censor
# y,H = censor(y,H,0.1,T-1)

# #-------------------------------------------------------------------------
# fltr = Filter(y,T,F,H,Q,R,m0,P0)
# fltr.Kalman(True)
# fltr.plot([a,b])

# fltr.Particle()
# fltr.plot([a,b])

#-------------------------------------------------------------------------
#GP MODEL
#-------------------------------------------------------------------------

# day = 1/360
# tau = day * 1
# T = 400 #Number of discrete data points
# dt = tau/T
# PQ = 0.03 #Probability of observing quote for each i = 0,1,...,N

# Ψ = 0.02 #Average half-spread
# A = 1 #Untransformed spread process mean reversion speed
# V = 20 #Untransformed spread process std dev
# σ = 0.4 #Bond YtB process std dev
# ς = 2*Ψ*0.05 #Measurement std dev

# paths = gp(Ψ,A,V,σ,ς,T,dt,PQ)

# x_ = np.array([paths[0],paths[1]]).T
# y = np.array(paths[2])
# J = paths[3]

# #-------------------------------------------------------------------------

# nx = 2
# N = 1000 #Number of particles
# c = 1

# Γ = (V**2 / (2*A)) * (1 - np.exp(-2 * A * dt))  #State 1 noise cov
# Σ = σ**2*dt                                     #State 2 noise cov
# R = ς**2                                        #Measurement noise cov
                                        
# m = np.zeros((T,nx))
# P = np.zeros((T,nx,nx))

# m[0] = x_[0]
# P[0] = 0.001 * np.eye(nx)

# F = np.array([[np.exp(-A*dt),0],
#               [0,            1]])
# Q = np.array([[Γ,0],
#               [0,Σ]])


# m0 = 0.1
# P0 = 0.001 * np.eye(nx)

# #-------------------------------------------------------------------------
# fltr = Filter(y,T,F,None,Q,R,m0,P0)
# fltr.GP(J, Ψ)

# fltr.plot_GP(x_)


















