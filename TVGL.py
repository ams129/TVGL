import numpy as np
import copy
import scipy as sp
import seaborn as sns

def gen_s(data, slice_size):
    S = []
    for i in range(0, data.shape[0] - slice_size + 1, slice_size):
        data_ = data[i:i + slice_size, :]
        data_ = sp.stats.zscore(data_, axis=0)
        s = np.cov(data_.T)
        S.append(np.matrix(s))
    return S

def initialize_theta(S):
    Theta = []
    Theta_ = np.zeros((S[0].shape[0], S[0].shape[1]))
    for i in range(len(S)):
        Theta.append(Theta_)
    return Theta

def initialize_z(S):
    Z0 = []
    Z1 = []
    Z2 = []
    Z_ = np.zeros((S[0].shape[0], S[0].shape[1]))
    for i in range(len(S)):
        Z0.append(Z_)
        Z1.append(Z_)
        Z2.append(Z_)
    return Z0, Z1, Z2

def initialize_u(S):
    U0 = []
    U1 = []
    U2 = []
    U_ = np.zeros((S[0].shape[0], S[0].shape[1]))
    for i in range(len(S)):
        U0.append(U_)
        U1.append(U_)
        U2.append(U_)
    return U0, U1, U2

def update_theta(slice_size, S, rho, alpha, beta, Theta, Z0, Z1, Z2, U0, U1, U2):
    #update Theta
    eta_ = 3 * rho / slice_size
    for i in range(len(Theta)):
        A = Z0[i] + Z1[i] + Z2[i] - U0[i] - U1[i] - U2[i]
        if i == 0 or i == len(Theta) - 1:
            A = (1 / 2) * A
        else:
            A = (1 / 3) * A
        d, q = np.linalg.eigh(eta_ * (1 / 2) * (A + A.T) - S[i])
        d = np.diag(d)
        Theta[i] = ( 1 / (2 * eta_)) * q * (d + np.sqrt(d ** 2 + 4 * eta_ * np.eye(A.shape[0]))) * q.T

    return Theta

def update_z_l1(slice_size, S, rho, alpha, beta, Theta, Z0, Z1, Z2, U0, U1, U2):
    #update Z_0
    for i in range(len(Z0)):
        A = Theta[i] + U0[i]
        for m in range(A.shape[0]):
            for n in range(m + 1, A.shape[1]):
                if abs(A[m, n]) <= alpha / rho:
                    A[m, n] = 0
                    A[n, m] = 0
                else:
                    if A[m, n] > 0:
                        A[m ,n] = A[m, n] - alpha / rho
                        A[n ,m] = A[n, m] - alpha / rho
                    else:
                        A[m, n] = A[m, n] + alpha / rho
                        A[n, m] = A[n, m] + alpha / rho
        Z0[i] = A

    #update Z_1, Z_2
    eta = 2 * beta / rho
    for i in range(1, len(Z1)):
        A = Theta[i] - Theta[i - 1] + U2[i] - U1[i - 1]
        for m in range(A.shape[0]):
            for n in range(m, A.shape[1]):
                if abs(A[m, n]) <= eta:
                    A[m, n] = 0
                    A[n, m] = 0
                else:
                    if A[m, n] > 0:
                        A[m, n] = A[m, n] - eta
                        A[n, m] = A[n, m] - eta
                    else:
                        A[m, n] = A[m, n] + eta
                        A[n, m] = A[n, m] + eta
        Z1[i - 1] = (1 / 2) * (Theta[i - 1] + Theta[i] + U1[i - 1] + U2[i]) - (1 / 2) * A
        Z2[i] = (1 / 2) * (Theta[i - 1] + Theta[i] + U1[i - 1] + U2[i]) + (1 / 2) * A

    return Z0, Z1, Z2

def update_z_l2(slice_size, S, rho, alpha, beta, Theta, Z0, Z1, Z2, U0, U1, U2):
    #update Z_0
    for i in range(len(Z0)):
        A = Theta[i] + U0[i]
        for m in range(A.shape[0]):
            for n in range(m + 1, A.shape[1]):
                if abs(A[m, n]) <= alpha / rho:
                    A[m, n] = 0
                    A[n, m] = 0
                else:
                    if A[m, n] > 0:
                        A[m ,n] = A[m, n] - alpha / rho
                        A[n ,m] = A[n, m] - alpha / rho
                    else:
                        A[m, n] = A[m, n] + alpha / rho
                        A[n, m] = A[n, m] + alpha / rho
        Z0[i] = A

    #update Z_1, Z_2
    eta = 2 * beta / rho
    for i in range(1, len(Z1)):
        A = Theta[i] - Theta[i - 1] + U2[i] - U1[i - 1]
        for m in range(A.shape[0]):
            for n in range(m, A.shape[1]):
                A[m, n] = (1 / (1 + 2 * eta)) * A[m, n]
                A[n, m] = (1 / (1 + 2 * eta)) * A[n, m]
        Z1[i - 1] = (1 / 2) * (Theta[i - 1] + Theta[i] + U1[i - 1] + U2[i]) - (1 / 2) * A
        Z2[i] = (1 / 2) * (Theta[i - 1] + Theta[i] + U1[i - 1] + U2[i]) + (1 / 2) * A

    return Z0, Z1, Z2

def update_u(slice_size, S, rho, alpha, beta, Theta, Z0, Z1, Z2, U0, U1, U2):
    #update U_0
    for i in range(len(S)):
        U0[i] = U0[i] + Theta[i] - Z0[i]

    #update U_1, U_2
    for i in range(1, len(S)):
        U1[i - 1] = U1[i - 1] + Theta[i - 1] - Z1[i - 1]
        U2[i] = U2[i] + Theta[i] - Z2[i]

    return U0, U1, U2

def calc_list_norm(A):
    norm = 0
    for i in range(len(A)):
        norm_ = 0
        for m in range(A[i].shape[0]):
            for n in range(m, A[i].shape[1]):
                norm_ += A[i][m,n] ** 2
        norm += norm_
    norm = np.sqrt(norm)

    return norm

def check_stop(rho, e_abs, e_rel, Theta, Z0, Z0_pre, U0):
    #calc e_dual and e_dual
    p = len(Theta) * ((Theta[0].shape[0] ** 2 - Theta[0].shape[0]) / 2 + Theta[0].shape[0])
    e_pri = np.sqrt(p) * e_abs + e_rel * max(calc_list_norm(Theta), calc_list_norm(Z0)) + .0001
    e_dual = np.sqrt(p) * e_abs + e_rel * calc_list_norm(U0) + .0001

    #calc res_pri
    res_pri = 0
    for i in range(len(Theta)):
        A = Theta[i] - Z0[i]
        norm = 0
        for m in range(A.shape[0]):
            for n in range(m, A.shape[1]):
                norm += A[m,n] ** 2
        res_pri += norm
    res_pri = np.sqrt(res_pri)

    #calc res_dual
    res_dual = 0
    for i in range(len(Z0)):
        A = rho * (Z0[i] - Z0_pre[i])
        norm = 0
        for m in range(A.shape[0]):
            for n in range(m, A.shape[1]):
                norm += A[m,n] ** 2
        res_dual += norm
    res_dual = np.sqrt(res_pri)

    return (res_pri < e_pri) and (res_dual < e_dual)

def admm(slice_size, S, rho, alpha, beta, penalty,  max_iters, e_abs, e_rel):
    #L1 penalty or L2 penalty
    if penalty == "L1":
        update_z = update_z_l1
    else:
        update_z = update_z_l2

    #initialize Theta, Z, U
    Theta = initialize_theta(S)
    Z0, Z1, Z2 = initialize_z(S)
    U0, U1, U2 = initialize_u(S)

    #start ADMM
    iters = 0
    stop = False
    while iters < max_iters:
        Z_pre = copy.deepcopy(Z0)
        Theta = update_theta(slice_size, S, rho, alpha, beta, Theta, Z0, Z1, Z2, U0, U1, U2)
        Z0, Z1, Z2 = update_z(slice_size, S, rho, alpha, beta, Theta, Z0, Z1, Z2, U0, U1, U2)
        U0, U1, U2 = update_u(slice_size, S, rho, alpha, beta, Theta, Z0, Z1, Z2, U0, U1, U2)
        iters = iters + 1
        if check_stop(rho, e_abs, e_rel, Theta, Z0, Z_pre, U0) == True:
            break
    if iters < max_iters:
        print("stop:", iters)
    else:
        print("max iters", iters)

    return Theta

def solve(data, alpha, beta, penalty, slice_size):
    #parameters
    rho = 1
    max_iters = 1e4
    e_abs = 1e-4
    e_rel = 1e-4

    #calc S
    S = gen_s(data, slice_size)

    #start solver
    print("alpha:", alpha, "beta:", beta, "slice_size:", slice_size, "S_length:", len(S))
    Theta = admm(slice_size, S, rho, alpha, beta, penalty, max_iters, e_abs, e_rel)

    return Theta, S
