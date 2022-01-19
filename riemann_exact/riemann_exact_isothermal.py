import numpy as np
import scipy.optimize as sco


def xi_equation(x, rho_l, rho_r):
    """
    The implicit equation for the isothermal shock Mach number of the isothermal Riemann problem.
    :param x: Shock Mach number
    :param rho_l: distant left density
    :param rho_r: distant right density
    :return:
    """

    return x * x * np.exp(x - 1./x) - rho_l / rho_r


def jac_xi_equation(x, rho_l, rho_r):
    """
    Jacobian (first derivative w.r.t. to x) of the implicit equation for the isothermal shock Mach number of the
    isothermal Riemann problem.
    :param x: Shock Mach number
    :param rho_l: distant left density (not used)
    :param rho_r: distant right density (not used)
    :return: Value of the first derivative
    """

    return 2 * x * np.exp(x - 1./x) + x * x * np.exp(x - 1./x) * (1. + 1. / (x * x))


def sol_rho(xi, grid, u, time, rho_l, rho_r):
    """
    Construct solution for rho at time t, given, xi, the solution for the speed u, on grid g
    :param xi: Shock mach number
    :param grid: Input grid with x-coordinates
    :param u: Solution of the speed on grid
    :param time: Time at which solution is sought
    :param rho_l: distant left density
    :param rho_r: distant right density
    :return: Solution of density on grid
    """

    # Boolean arrays marking regions
    region_2, region_l, region_r, region_f = identify_regions(grid, time, xi)

    # Construct rho solution
    rho = np.zeros_like(grid)
    mach = u
    rho[region_r] = rho_r
    rho[region_l] = rho_l
    rho[region_2] = rho_r * xi * xi
    rho[region_f] = rho_l * np.exp(-mach[region_f])

    return rho


def sol_u(xi, grid, time, u_l, u_r):
    """
    Construct solution for rho at time t, given, xi, the solution for the speed u, on grid g
    :param xi: Shock mach number
    :param grid: Input grid with x-coordinates
    :param time: Time at which solution is sought
    :param u_l: distant left speed
    :param u_r: distant right speed
    :return: Solution of speed on grid
    """
    # Boolean arrays marking regions
    region_2, region_l, region_r, region_rf = identify_regions(grid, time, xi)

    # Construct solutions
    u = np.zeros_like(grid)
    u[region_r] = u_r
    u[region_l] = u_l
    u[region_2] = xi - 1. / xi
    u[region_rf] = (grid[region_rf] + time) / time

    return u


def identify_regions(grid, time, xi):
    """
    Identify regions in the solution. Distant right state (region_r) distant left state (region_l),
    postshock region (region_2), rarefaction wave (region_f).
    :param grid: grid - a 1D array of position values
    :param time: Time at which solution and regions are sought
    :param xi: Shock mach number
    :return: Four arrays (region_2, region_l, region_r, region_f) of the same size as grid with boolean values
    indicating whether the position value on the grid belongs to the corresponding region or not.
    These arrays can be used as slices for other arrays.
    """

    # Distant right and left regions
    region_r = grid - xi * time > 0
    region_l = grid + time <= 0

    # Postshock region
    region_2r = grid - xi * time <= 0
    region_2l = grid - (xi - 1. / xi - 1.) * time > 0
    region_2 = np.logical_and(region_2r, region_2l)

    # Rarefaction wave
    region_fr = grid - (xi - 1. / xi - 1.) * time <= 0
    region_fl = grid + time > 0
    region_f = np.logical_and(region_fr, region_fl)

    return region_2, region_l, region_r, region_f


def riemann_exact(ql, qr, grid, time):
    """
    Solves the Riemann problem for an isothermal gas.
    :param ql: left state vector (rho_l, u_l).
    :param qr: right state vector (rho_r, u_r).
    :param grid: grid on which to evaluate solution. Initial discontinuity is assumed to be at x=0.
    :param time: time at which solution is desired.
    :return: Solution on grid. Same size as grid.
    """

    if time > 0:
        rho_l, rho_r = ql[0], qr[0]
        u_l, u_r = ql[1], qr[1]
        guess = np.array([1.])

        res = sco.newton(xi_equation, guess, args=(rho_l, rho_r))
        print(f'Root found: xi = {res}')

        u = sol_u(res, grid, time, u_l, u_r)
        rho = sol_rho(res, grid, u, time, rho_l, rho_r)

    else:
        rho = np.where(grid < 0, ql[0], qr[0])
        u = np.where(grid < 0, ql[1], qr[1])

    return rho, u
