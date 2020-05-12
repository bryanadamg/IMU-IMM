import numpy as np
import matplotlib.pyplot as plt
import math
from usv_class import usv

class ca(usv):

    # Constants
    H = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1]])
    Q = np.array([[0.0001, 0, 0, 0, 0],
                    [0, 0.0001, 0, 0, 0],
                    [0, 0, 0.05, 0, 0],
                    [0, 0, 0, 0.05, 0],
                    [0, 0, 0, 0, math.radians(0.005)]])
    R = np.array([[usv.gps_sd**2, 0, 0],
                    [0, usv.gps_sd**2, 0],
                    [0, 0, usv.bearing_sd**2]])


    def __init__(self, xs, the_a = 0.6, the_b = 2, the_k = -2):
        self.xs = xs
        self.a, self.b, self.k = the_a, the_b, the_k
        self.x_hat = xs
        self.P = np.array([[usv.gps_sd, 0, 0, 0, 0],
                            [0, usv.gps_sd, 0, 0, 0],
                            [0, 0, 0.5, 0, 0],
                            [0, 0, 0, 0.5, 0],
                            [0, 0, 0, 0, usv.bearing_sd]])

    @staticmethod
    def const_acc_pred(xs,abx,aby,omega,dt):
        px = xs[0] + dt*xs[2] + 0.5*(dt**2)*(math.cos(-xs[4])*abx - math.sin(-xs[4])*aby)
        py = xs[1] + dt*xs[3] + 0.5*(dt**2)*(math.sin(-xs[4])*abx + math.cos(-xs[4])*aby)
        # navigational frame velocity
        vx = xs[2] + dt*(math.cos(-xs[4])*abx - math.sin(-xs[4])*aby)
        vy = xs[3] + dt*(math.sin(-xs[4])*abx + math.cos(-xs[4])*aby)
        theta = xs[4] + dt*omega
        
        return np.array([px, py, vx, vy, theta])

    def sigma(self, mu, cov):
        n = len(mu)
        lamb = self.a**2*(n+self.k)-n

        # Calculate Weights
        Wc = np.full(2*n+1, 1/(2*(n+lamb)))
        Wc[0] = lamb/(n+lamb)+1-self.a**2+self.b
        Wm = np.full(2*n+1, 1/(2*(n+lamb)))
        Wm[0] = lamb/(n+lamb)

        # Generate Sigma points
        # Transpose to store vertically
        X = np.zeros((2*n+1,n))
        X[0] = mu.T

        U = np.linalg.cholesky(np.dot((n+lamb),cov))
        for i in range(n):
            X[i+1] = mu.T + U[i]
            X[i+n+1] = mu.T - U[i]

        return X, Wm, Wc

    # Propagate sigma points through a function
    @staticmethod
    def propagate(X, abx, aby, omega, dt, const_a_function):
        n = np.shape(X)[1]
        result = np.zeros((2*n+1,n))
        for i in range(len(X)):
            result[i] = const_a_function(X[i],abx,aby,omega,dt)
        return result

    # Y is a (2n+1,n) array
    # mu is a (1,n) horizontal array
    @staticmethod
    def recover(sigma, Wm, Wc, cov):
        n = np.shape(sigma)[1]
        mu = np.zeros(n)
        for i in range(len(sigma)):
            mu += Wm[i] * sigma[i]
        P = np.zeros((n,n))
        for i in range(len(sigma)):
            residual = np.array([sigma[i]] - mu)
            P += Wc[i] * residual.T * residual
        P += cov
        
        return np.array([mu]), P

    # y is residual in matrix form
    # S is system covariance in matrix form
    @property
    def likelihood(self):
        return 1/math.sqrt(2*math.pi*np.linalg.det(self.S)) * math.exp(-0.5*self.r.T.dot(np.linalg.inv(self.S)).dot(self.r))

    def predict(self, imu):
        X, Wm, Wc = self.sigma(self.xs, self.P)
        Y = self.propagate(X, imu[0], imu[1], imu[2], usv.dt, self.const_acc_pred)
        self.x_hat, self.P_hat = self.recover(Y, Wm, Wc, self.Q)

    def update(self, z):
        self.r = z - self.H.dot(self.x_hat.T)
        self.S = (self.H.dot(self.P_hat)).dot(self.H.T) + self.R
        K = (self.P_hat.dot(self.H.T)).dot(np.linalg.inv(self.S))
        # Estimate
        self.xs = self.x_hat.T + K.dot(self.r)
        self.P = (np.eye(len(self.xs)) - K.dot(self.H)).dot(self.P_hat)

class cv(usv):

    # Constants
    H = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1]])
    Q = np.array([[0.0000001, 0, 0, 0, 0],
                    [0, 0.0000001, 0, 0, 0],
                    [0, 0, 0.001, 0, 0],
                    [0, 0, 0, 0.001, 0],
                    [0, 0, 0, 0, math.radians(0.002)]])
    R = np.array([[usv.gps_sd**2, 0, 0],
                    [0, usv.gps_sd**2, 0],
                    [0, 0, usv.bearing_sd**2]])


    def __init__(self, xs, the_a = 0.6, the_b = 2, the_k = -2):
        self.xs = xs
        self.a, self.b, self.k = the_a, the_b, the_k
        self.x_hat = xs
        self.P = np.array([[usv.gps_sd, 0, 0, 0, 0],
                            [0, usv.gps_sd, 0, 0, 0],
                            [0, 0, 0.5, 0, 0],
                            [0, 0, 0, 0.5, 0],
                            [0, 0, 0, 0, usv.bearing_sd]])

    @staticmethod
    def const_v_pred(xs, dt):
        px = xs[0] + dt*xs[2]
        py = xs[1] + dt*xs[3]
        # navigational frame velocity
        vx = xs[2]
        vy = xs[3]
        theta = xs[4]
        return np.array([px, py, vx, vy, theta])

    def sigma(self, mu, cov):
        n = len(mu)
        lamb = self.a**2*(n+self.k)-n

        # Calculate Weights
        Wc = np.full(2*n+1, 1/(2*(n+lamb)))
        Wc[0] = lamb/(n+lamb)+1-self.a**2+self.b
        Wm = np.full(2*n+1, 1/(2*(n+lamb)))
        Wm[0] = lamb/(n+lamb)

        # Generate Sigma points
        # Transpose to store vertically
        X = np.zeros((2*n+1,n))
        X[0] = mu.T

        U = np.linalg.cholesky(np.dot((n+lamb),cov))
        for i in range(n):
            X[i+1] = mu.T + U[i]
            X[i+n+1] = mu.T - U[i]

        return X, Wm, Wc

    # Propagate sigma points through a function
    @staticmethod
    def propagate(X, dt, function):
        n = np.shape(X)[1]
        result = np.zeros((2*n+1,n))
        for i in range(len(X)):
            result[i] = function(X[i],dt)
        return result

    # Y is a (2n+1,n) array
    # mu is a (1,n) horizontal array
    @staticmethod
    def recover(sigma, Wm, Wc, cov):
        n = np.shape(sigma)[1]
        mu = np.zeros(n)
        for i in range(len(sigma)):
            mu += Wm[i] * sigma[i]
        P = np.zeros((n,n))
        for i in range(len(sigma)):
            residual = np.array([sigma[i]] - mu)
            P += Wc[i] * residual.T * residual
        P += cov
        
        return np.array([mu]), P

    # y is residual in matrix form
    # S is system covariance in matrix form
    @property
    def likelihood(self):
        return 1/math.sqrt(2*math.pi*np.linalg.det(self.S)) * math.exp(-0.5*self.r.T.dot(np.linalg.inv(self.S)).dot(self.r))

    def predict(self, imu):
        X, Wm, Wc = self.sigma(self.xs, self.P)
        Y = self.propagate(X, usv.dt, self.const_v_pred)
        self.x_hat, self.P_hat = self.recover(Y, Wm, Wc, self.Q)

    def update(self, z):
        self.r = z - self.H.dot(self.x_hat.T)
        self.S = (self.H.dot(self.P_hat)).dot(self.H.T) + self.R
        K = (self.P_hat.dot(self.H.T)).dot(np.linalg.inv(self.S))
        # Estimate
        self.xs = self.x_hat.T + K.dot(self.r)
        self.P = (np.eye(len(self.xs)) - K.dot(self.H)).dot(self.P_hat)


class ct(usv):

    # Constants
    H = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1]])
    Q = np.array([[0.00001, 0, 0, 0, 0],
                    [0, 0.00001, 0, 0, 0],
                    [0, 0, 0.01, 0, 0],
                    [0, 0, 0, 0.01, 0],
                    [0, 0, 0, 0, math.radians(0.5)]])
    R = np.array([[usv.gps_sd**2, 0, 0],
                    [0, usv.gps_sd**2, 0],
                    [0, 0, usv.bearing_sd**2]])

    def __init__(self, xs, the_a = 0.6, the_b = 2, the_k = -2):
        self.xs = xs
        self.a, self.b, self.k = the_a, the_b, the_k
        self.x_hat = xs
        self.P = np.array([[usv.gps_sd, 0, 0, 0, 0],
                            [0, usv.gps_sd, 0, 0, 0],
                            [0, 0, 0.5, 0, 0],
                            [0, 0, 0, 0.5, 0],
                            [0, 0, 0, 0, usv.bearing_sd]])

    @staticmethod
    def const_t_pred(xs, omega, dt):
        px = xs[0] + math.sin(omega*dt)*xs[2]/omega - (1-math.cos(omega*dt))*xs[3]/omega
        py = xs[1] + (1-math.cos(omega*dt))*xs[2]/omega + math.sin(omega*dt)*xs[3]/omega
        # navigational frame velocity
        vx = math.cos(omega*dt)*xs[2] - math.sin(omega*dt)*xs[3]
        vy = math.sin(omega*dt)*xs[2] + math.cos(omega*dt)*xs[3]
        theta = xs[4] + omega*dt
        return np.array([px, py, vx, vy, theta])

    def sigma(self, mu, cov):
        n = len(mu)
        lamb = self.a**2*(n+self.k)-n

        # Calculate Weights
        Wc = np.full(2*n+1, 1/(2*(n+lamb)))
        Wc[0] = lamb/(n+lamb)+1-self.a**2+self.b
        Wm = np.full(2*n+1, 1/(2*(n+lamb)))
        Wm[0] = lamb/(n+lamb)

        # Generate Sigma points
        # Transpose to store vertically
        X = np.zeros((2*n+1,n))
        X[0] = mu.T

        U = np.linalg.cholesky(np.dot((n+lamb),cov))
        for i in range(n):
            X[i+1] = mu.T + U[i]
            X[i+n+1] = mu.T - U[i]

        return X, Wm, Wc

    @staticmethod
    def propagate(X, omega, dt, function):
        n = np.shape(X)[1]
        result = np.zeros((2*n+1, n))
        for i in range(len(X)):
            result[i] = function(X[i], omega, dt)
        return result

    @staticmethod
    def recover(sigma, Wm, Wc, cov):
        n = np.shape(sigma)[1]
        mu = np.zeros(n)
        for i in range(len(sigma)):
            mu += Wm[i] * sigma[i]
        P = np.zeros((n,n))
        for i in range(len(sigma)):
            residual = np.array([sigma[i]] - mu)
            P += Wc[i] * residual.T * residual
        P += cov
        
        return np.array([mu]), P

    @property
    def likelihood(self):
        return 1/math.sqrt(2*math.pi*np.linalg.det(self.S)) * math.exp(-0.5*self.r.T.dot(np.linalg.inv(self.S)).dot(self.r))

    def predict(self, imu):
        X, Wm, Wc = self.sigma(self.xs, self.P)
        Y = self.propagate(X, -imu[2], usv.dt, self.const_t_pred)
        self.x_hat, self.P_hat = self.recover(Y, Wm, Wc, self.Q)

    def update(self, z):
        self.r = z - self.H.dot(self.x_hat.T)
        self.S = (self.H.dot(self.P_hat)).dot(self.H.T) + self.R
        K = (self.P_hat.dot(self.H.T)).dot(np.linalg.inv(self.S))
        # Estimate
        self.xs = self.x_hat.T + K.dot(self.r)
        self.P = (np.eye(len(self.xs)) - K.dot(self.H)).dot(self.P_hat)


class imm(usv):

    def __init__(self, xs, the_models, the_M, the_mode):
        self.xs = xs
        self.M = the_M
        self.models = the_models    # list of models
        self.mode = the_mode
        self.n = len(self.models)    # number of models
        self.k = np.shape(self.xs)[0]   # model dimension
        self.omega = np.zeros((self.n, self.n))


    def initialize(self):

        self.cbar = self.mode.dot(self.M)
        for i in range(self.n):
            for j in range(self.n):
                self.omega[i, j] = (self.M[i, j] * self.mode[0, i]) / self.cbar[0, j]

        x0, P0 = [0] * self.n, [0] * self.n
        for i in range(self.n):
            for j in range(self.n):
                x0[i] += self.models[j].xs * self.omega[j, i]
        for i in range(self.n):
            for j in range(self.n):
                diff = self.models[i].xs - x0[j]
                P0[j] += self.omega[i, j] * (self.models[i].P + np.outer(diff, diff))

        for i in range(len(self.models)):
            self.models[i].xs = x0[i]
            self.models[i].P = P0[i]

    def predict(self, imu):
        # PREDICTION
        for model in self.models:
            model.predict(imu)

    # MODEL PROBABILITY UPDATE
    def update(self, z):

        for model in self.models:
            model.update(z)

        for i in range(self.n):
            self.mode[0, i] = self.cbar[0, i] * self.models[i].likelihood
        # print(np.sum(self.mode))
        self.mode /= np.sum(self.mode)
    
    def fuse(self):

        self.xs_imm, self.P_imm = np.zeros((self.k, 1)), np.zeros((self.k, self.k))
        for i in range(len(self.models)):
            self.xs_imm += self.models[i].xs * self.mode[0, i]
        for i in range(len(self.models)):
            diff = self.models[i].xs - self.xs_imm
            self.P_imm += self.mode[0, i] * (self.models[i].P + np.outer(diff, diff))

        for model in self.models:
            model.xs = np.copy(self.xs_imm)
            model.P = np.copy(self.P_imm)


if __name__ == "__main__":
    # Create object and initialize values
    usv1 = usv([0,0],[0,1.5],0)
    xa = np.array([[usv1.pox, usv1.poy, usv1.vox, usv1.voy, usv1.heading]]).T
    ca1 = ca(xa)
    cv1 = cv(xa)

    usv1.cmd_heading(60, math.radians(20))   
