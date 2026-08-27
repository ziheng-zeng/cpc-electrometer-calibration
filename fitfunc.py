import numpy as np


def sigmoid(x, eta, dp_50, k, a, c):
    return eta / (1 + np.exp(-k * (x - dp_50))) - np.exp(-a * x + c)


def hill_langmuir_loss(x, k_a, n, loss, q):
    return 1 / (1 + (k_a / x) ** n) - np.exp(-loss * x) + q


def mu_g(T_degC=20):
    """Returns the air viscosity [kg/m-s]"""
    T_K = T_degC + 273.15
    mu_g_r = 1.83 * 10**-5  # reference gas viscosity [kg/m-s]
    T_r = 296.15  # reference temperature [K]
    P_r = 101.3  # reference pressure [kPa]
    Suth = 110.4  # Sutherland const. [K]
    mu = mu_g_r * (T_r + Suth) / (T_K + Suth) * (T_K / T_r) ** (3 / 2)
    return mu


def lambda_mfp(T_degC=20, P_kPa=101.3):
    """Returns the gas mean free path [m]"""
    T_K = T_degC + 273.15
    lambda_mfp_r = 6.73 * 10**-8  # reference mean free path [m]
    T_r = 296.15  # reference temperature [K]
    P_r = 101.3  # reference pressure [kPa]
    Suth = 110.4  # Sutherland const. [K]

    lam = (
        lambda_mfp_r
        * (P_r / P_kPa)
        * (T_K / T_r)
        * (1 + Suth / T_r)
        / (1 + Suth / T_K)
    )
    return lam


def Kn(Dp, T_degC=20, P_kPa=101.3):
    """Returns the Knudsen number"""
    Dp = np.array([Dp]).flatten()
    Dp_SI = Dp * 1e-9
    Kn = 2 * lambda_mfp(T_degC, P_kPa) / Dp_SI

    return Kn if len(Kn) > 1 else Kn[0]


def Cc(Dp, T_degC=20, P_kPa=101.3):
    """Returns the Cunningham slip correction factor"""
    Cc_alpha, Cc_beta, Cc_gamma = 1.142, 0.558, 0.999  # Cc constants
    Kn_num = Kn(Dp, T_degC, P_kPa)
    Cc = 1 + Kn_num * (Cc_alpha + Cc_beta * np.exp(-Cc_gamma / Kn_num))

    if np.isscalar(Cc):
        return Cc
    else:
        return Cc if len(Cc) > 1 else Cc.item()


def GK_eta(Dp, L_tube, Q_lpm=0.3, T_degC=20, P_kPa=101.3):
    """Gormley-Kennedy (1949) particle transmission efficiency
    in laminar flow through a tube"""
    T_K = T_degC + 273.15
    # Dp = np.array([Dp]).flatten()
    Dp_SI = Dp * 1e-9
    k_Bltz = 1.380658e-23  # Boltzmann constant [J/K]
    Dc = (
        k_Bltz
        * T_K
        * Cc(Dp, T_degC, P_kPa)
        / (3 * np.pi * mu_g(T_degC) * Dp_SI)
    )  # Diffusion coefficient [m^2/s]
    Q_SI = Q_lpm * 0.001 / 60  # flowrate [m^3/s]
    xi = Dc * (L_tube * np.pi / Q_SI)

    # next lines calculate eta for xi<0.02 or xi>0.02
    eta_xi_lt_p02 = (
        1 - 2.5638 * xi ** (2 / 3) + 1.2 * xi + 0.1767 * xi ** (4 / 3)
    )  # xi<=0.02
    eta_xi_gt_p02 = (
        0.81905 * np.exp(-3.6568 * xi)
        + 0.09753 * np.exp(-22.305 * xi)
        + 0.0325 * np.exp(-56.961 * xi)
        + 0.01544 * np.exp(-107.62 * xi)
    )  # xi>0.02
    eta = eta_xi_lt_p02 * (xi <= 0.02) + eta_xi_gt_p02 * (xi > 0.02)

    return eta  # if len(eta) > 1 else eta[0]


def cpc_eta_activation(x, eta, d50, d0):
    """Returns CPC activation efficiency according to Stolzenburg & McMurry (1991)
    formulation (ultrafine-CPC)"""
    x = np.array([x]).flatten()
    y = eta * (1 - np.exp(-np.log(2) * (x - d0) / (d50 - d0)))
    y[y < 0] = 0

    return y  # if len(y) > 1 else y[0]


def cpc_eta_activ_w_GK(x, eta, d50, d0, L=0.05, Q=0.3, T_degC=20, P_kPa=101.3):
    """Returns CPC activation efficiency multiplied with
    Gormley-Kennedy transmission efficiency"""
    y = cpc_eta_activation(x, eta, d50, d0) * GK_eta(x, L, Q, T_degC, P_kPa)
    y[y < 0] = 0

    return y  # if len(y) > 1 else y[0]
