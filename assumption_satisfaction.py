import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt

class arx(): # y_t = [y_t-1 ... y_t-na]'a + [u_t-1 ... u_t-nb]'b
    def __init__(self, a: np.ndarray , b: np.ndarray, phi_t :np.ndarray|None = None, y_t:np.ndarray|None =None):
        self.a = a
        self.b = b
        self.na = a.shape[0]
        self.nb = b.shape[0]

        if phi_t is None: #regressor at time t
            self.phi_t = np.zeros(self.na+self.nb) # if not specified it starts from all zeros
        else: 
            self.phi_t = phi_t
        if y_t is None: # if not specified it starts from zero
            self.y_t = 0.
        else: 
            self.y_t = y_t

    def step(self,u_t = None, e_t = None):
        if u_t is None: # free dynamics if not specified
            u_t = 0
        if e_t is None: #no error if not specified
            e_t = 0

        y_t = self.y_t
        self.update_phi_t(y_t,u_t) # the regressor is updated with the current value of y_t and the given input u_t

        y = self.a@self.phi_t[:self.na] + self.b @ self.phi_t[self.na:] + e_t #calculates value of y_t
        self.y_t = y #assigns it to the system model

    def get_yt(self):
        return self.y_t
        
    def get_yt_phit(self):
        return self.y_t, self.phi_t

    def update_phi_t(self,y_t,u_t):
        phi_y = self.phi_t[:self.na]  #this divides the regressor in two, one with the past values of y, one with the past values of u
        phi_u = self.phi_t[self.na:]

        phi_y = np.roll(phi_y,1) # for example the vector [0 1 2 3] becomes [3 0 1 2]
        phi_y[0] = y_t
        phi_u = np.roll(phi_u,1)
        phi_u[0] = u_t

        self.phi_t = np.concatenate([phi_y,phi_u],0)


class estimator():
    def __init__(self,lamb, n: int|None = None, W0: np.ndarray|None = None, time_varying: bool = False):
        if W0 is None and n is not None:
            self.Wt = np.eye(n)
        if W0 is not None:
            self.Wt = W0
            n = self.Wt.shape[0]
        self.Pt = np.linalg.pinv(self.Wt,hermitian=True)
        self.lamb = lamb
        self.theta_hat = np.zeros(n)
        self.time_varying = time_varying

    def update_theta(self,yt,xt):
        theta = self.theta_hat
        lamb = self.lamb
        Pt = self.Pt
        eps = yt - xt@theta
        arg = lamb*eps/(xt@Pt@xt)
        sat_1 = lambda x: np.clip(x,-1,1)  # one-line function definitions
        theta = theta + 1/lamb*sat_1(arg)*Pt@xt
        self.theta_hat = theta
        if self.time_varying:
            self.update_Wt_Pt(xt)

    def update_Wt_Pt(self,xt):
        Wt = self.Wt
        lamb = self.lamb
        xx = xt.reshape(-1,1)@xt.reshape(1,-1)
        Wt = lamb* Wt + xx
        Wt = Wt
        self.Wt = (Wt + Wt.T) /2
        Pt = self.Pt
        Pt = 1/lamb*(Pt - Pt@xx@Pt/(lamb+xt@Pt@xt))
        self.Pt = (Pt + Pt.T)/2

    def get_theta(self):
        return self.theta_hat

    def get_Wt_Pt(self):
        return self.Wt, self.Pt

if __name__ == '__main__':
    a = np.array([0.5,-0.7,0.1])
    b = np.array([1,-0.3,0])

    theta0 = np.concatenate([a,b])

    s1 = arx(a,b)
    W0 = np.eye(6)
    W0[3,3] = 100
    est = estimator(lamb = 0.99 ,n=6, W0 = W0, time_varying=True)

    N = 10000

    rng = np.random.default_rng()
    u = rng.uniform(-1,1,N)
    e = rng.standard_normal(N)*0
    y = np.zeros(N+1)
    norm_x2 = np.zeros(N)
    eig = np.zeros(N)
    theta = np.zeros((N+1,6))
    for i in range(N):
        s1.step(u[i],e[i])
        yt, xt = s1.get_yt_phit()
        est.update_theta(yt,xt)
        thetat = est.get_theta()

        theta[i+1,:] = thetat
        y[i+1] = yt

        Wt, _ = est.get_Wt_Pt()                   
        eig_t = np.linalg.eigvalsh(Wt)[0]
        norm_x2[i] = np.linalg.norm(xt,2)**2
        eig[i] = eig_t

    plt.plot(theta, 'b')
    plt.plot(np.array([0,N]),np.concatenate([theta0.reshape(1,-1),theta0.reshape(1,-1)]),'r')
    plt.show()

    plt.plot(norm_x2, label=r'$||x||_2^2$')
    plt.plot(eig, label = r'$\lambda_{min}$')
    plt.legend()
    plt.show()