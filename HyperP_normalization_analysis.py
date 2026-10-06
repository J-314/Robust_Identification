import numpy as np
import matplotlib.pyplot as plt

class arx(): 
    def __init__(self, a: np.ndarray , b: np.ndarray, phi_t: np.ndarray|None = None, y_t: np.ndarray|None = None):
        self.a = a
        self.b = b
        self.na = a.shape[0]
        self.nb = b.shape[0]
        self.phi_t = np.zeros(self.na + self.nb) if phi_t is None else phi_t
        self.y_t = 0. if y_t is None else y_t

    def step(self, u_t = 0, e_t = 0):
        self.update_phi_t(self.y_t, u_t) 
        self.y_t = self.a @ self.phi_t[:self.na] + self.b @ self.phi_t[self.na:] + e_t 

    def get_yt(self):
        return self.y_t
        
    def get_yt_phit(self):
        return self.y_t, self.phi_t

    def update_phi_t(self, y_t, u_t):
        phi_y = self.phi_t[:self.na]  
        phi_u = self.phi_t[self.na:]
        phi_y = np.roll(phi_y, 1) 
        phi_y[0] = y_t
        phi_u = np.roll(phi_u, 1)
        phi_u[0] = u_t
        self.phi_t = np.concatenate([phi_y, phi_u], 0)

class estimator():
    def __init__(self, lamb, n: int|None = None, W0: np.ndarray|None = None, update_type: str = 'standard'):
        if W0 is None and n is not None:
            self.Wt = np.eye(n)
        if W0 is not None:
            self.Wt = W0.copy()
            n = self.Wt.shape[0]
        self.Pt = np.linalg.pinv(self.Wt, hermitian=True)
        self.lamb = lamb
        self.theta_hat = np.zeros(n)
        self.update_type = update_type
        self.epsilon = 1e-4 

    def update_theta(self, yt, xt):
        theta = self.theta_hat
        eps = yt - xt @ theta
        
        if self.update_type in ['normalized', 'normalized_sq']:
            omega = 1.0 / (np.abs(eps) + self.epsilon) 
        else:
            omega = 1.0
            
        arg = self.lamb * eps / (xt @ self.Pt @ xt)
        sat_1 = lambda x: np.clip(x, -1, 1)  
        self.theta_hat = theta + (1 / self.lamb) * sat_1(arg) * self.Pt @ xt
        
        if self.update_type != 'constant':
            weight = omega**2 if self.update_type == 'normalized_sq' else omega
            self.update_Wt_Pt(xt, weight)

    def update_Wt_Pt(self, xt, weight=1.0):
        xx = xt.reshape(-1, 1) @ xt.reshape(1, -1)
        Wt = self.lamb * self.Wt + weight * xx
        self.Wt = (Wt + Wt.T) / 2
        
        Pt = self.Pt
        Pt = (1 / self.lamb) * (Pt - (weight * Pt @ xx @ Pt) / (self.lamb + weight * xt @ Pt @ xt))
        self.Pt = (Pt + Pt.T) / 2

    def get_theta(self):
        return self.theta_hat

def run_simulation(N, lamb, update_type, dwell_time, impulse_prob, impulse_amp_bound, a, b, amplitude_u, amplitude_e):
    theta0 = np.concatenate([a, b])
    sys = arx(a, b)
    rng = np.random.default_rng(42)
    
    W0 = np.eye(6)
    if update_type == 'constant':
        try:
            with np.load("Wt.npz") as data:
                W0 = data["Wt"]
        except FileNotFoundError:
            print("Warning: Wt.npz not found. Using Identity matrix for constant W.")
    
    est = estimator(lamb=lamb, n=6, W0=W0, update_type=update_type)
    
    u = rng.standard_normal(N) * amplitude_u
    e = rng.standard_normal(N) * amplitude_e
    
    f = np.zeros(N)
    time_since_last = dwell_time
    for i in range(N):
        if time_since_last >= dwell_time:
            if rng.random() < impulse_prob: 
                f[i] = rng.uniform(-impulse_amp_bound, impulse_amp_bound)
                time_since_last = 0
            else:
                time_since_last += 1
        else:
            time_since_last += 1
            
    noise = e + f
    theta_hist = np.zeros((N, 6))
    
    for i in range(N):
        sys.step(u[i], noise[i])
        yt, xt = sys.get_yt_phit()
        est.update_theta(yt, xt)
        theta_hist[i, :] = est.get_theta()
        
    mse = np.mean(np.linalg.norm(theta_hist - theta0, axis=1)**2)
    
    return theta_hist, mse

def plot_theta_trajectory(lamb, dwell_time, update_type, N=5000):
    a, b = np.array([0.5, -0.7, 0.1]), np.array([1, -0.3, 0])
    theta0 = np.concatenate([a, b])
    theta_hist, mse = run_simulation(N, lamb, update_type, dwell_time, 0.05, 50, a, b, 1, 0.05)
    
    plt.figure(figsize=(10, 5))
    plt.plot(theta_hist)
    for t_val in theta0:
        plt.axhline(t_val, color='red', linestyle='--', alpha=0.5)
    plt.title(f"Trajectory ({update_type}) | Lambda={lamb} | Dwell={dwell_time} | MSE={mse:.4f}")
    plt.xlabel("Steps")
    plt.ylabel("Theta parameters")
    plt.show()

if __name__ == '__main__':
    a = np.array([0.5, -0.7, 0.1])
    b = np.array([1, -0.3, 0])
    
    # 1. Trajectory Plotter
    plot_theta_trajectory(lamb=0.995, dwell_time=1000, update_type='constant')
    plot_theta_trajectory(lamb=0.995, dwell_time=1000, update_type='standard')
    plot_theta_trajectory(lamb=0.995, dwell_time=1000, update_type='normalized')
    plot_theta_trajectory(lamb=0.995, dwell_time=1000, update_type='normalized_sq')

    # 2. Dwell Time Analysis
    dwell_times = [50, 200, 500, 1000, 2000, 5000]
    lambda_val = 0.995
    mse_const, mse_std, mse_norm, mse_norm_sq = [], [], [], []
    
    for dt in dwell_times:
        _, m_c = run_simulation(5000, lambda_val, 'constant', dt, 0.05, 50, a, b, 1, 0.05)
        _, m_s = run_simulation(5000, lambda_val, 'standard', dt, 0.05, 50, a, b, 1, 0.05)
        _, m_n = run_simulation(5000, lambda_val, 'normalized', dt, 0.05, 50, a, b, 1, 0.05)
        _, m_n_sq = run_simulation(5000, lambda_val, 'normalized_sq', dt, 0.05, 50, a, b, 1, 0.05)
        mse_const.append(m_c)
        mse_std.append(m_s)
        mse_norm.append(m_n)
        mse_norm_sq.append(m_n_sq)

    plt.figure(figsize=(8, 5))
    plt.plot(dwell_times, mse_const, marker='^', color='blue', label='Constant W')
    plt.plot(dwell_times, mse_std, marker='o', color='red', label='Standard Update')
    plt.plot(dwell_times, mse_norm, marker='s', color='green', label=r'Normalized Update ($\omega$)')
    plt.plot(dwell_times, mse_norm_sq, marker='D', color='purple', label=r'Normalized Update ($\omega^2$)')
    plt.xlabel("Dwell Time")
    plt.ylabel("Parameter MSE (Log Scale)")
    plt.yscale('log')
    plt.title(r"Impact of Dwell Time ($\lambda=0.995$)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

    # 3. Lambda Analysis
    lambda_vals = [0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
    dt_fixed = 1000
    mse_l_const, mse_l_std, mse_l_norm, mse_l_norm_sq = [], [], [], []
    
    for l_val in lambda_vals:
        _, m_c = run_simulation(5000, l_val, 'constant', dt_fixed, 0.05, 50, a, b, 1, 0.05)
        _, m_s = run_simulation(5000, l_val, 'standard', dt_fixed, 0.05, 50, a, b, 1, 0.05)
        _, m_n = run_simulation(5000, l_val, 'normalized', dt_fixed, 0.05, 50, a, b, 1, 0.05)
        _, m_n_sq = run_simulation(5000, l_val, 'normalized_sq', dt_fixed, 0.05, 50, a, b, 1, 0.05)
        mse_l_const.append(m_c)
        mse_l_std.append(m_s)
        mse_l_norm.append(m_n)
        mse_l_norm_sq.append(m_n_sq)

    plt.figure(figsize=(8, 5))
    plt.plot(lambda_vals, mse_l_const, marker='^', color='blue', label='Constant W')
    plt.plot(lambda_vals, mse_l_std, marker='o', color='red', label='Standard Update')
    plt.plot(lambda_vals, mse_l_norm, marker='s', color='green', label=r'Normalized Update ($\omega$)')
    plt.plot(lambda_vals, mse_l_norm_sq, marker='D', color='purple', label=r'Normalized Update ($\omega^2$)')
    plt.xlabel("Lambda Value")
    plt.ylabel("Parameter MSE (Log Scale)")
    plt.yscale('log')
    plt.title(f"Impact of Lambda (Dwell Time={dt_fixed})")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()