import numpy as np
import scipy.optimize as sco


def f_shock(x, rho, prs, gamma):

    a = 2. / ((gamma + 1.) * rho)
    b = (gamma - 1.) / (gamma + 1.) * prs
    return (x - prs) * np.sqrt(a / (x + b))


def f_rare(x, rho, prs, gamma):

    a = np.sqrt(gamma * prs / rho)

    return (2. * a) / (gamma - 1.) * ((x / prs) ** ((gamma - 1.) / (2. * gamma)) - 1.)


def p_equation(x, rho_l, rho_r, u_l, u_r, prs_l, prs_r, gamma):
    """
    The implicit equation for the pressure shock Mach number of the adiabatic Riemann problem.
    :param x: pressure p
    :param rho_l: distant left density
    :param rho_r: distant right density
    :param u_l: distant left velocity
    :param u_r: distant right velocity
    :param prs_l: distant left pressure
    :param prs_r: distant right pressure
    :param gamma: adiabatic index
    :return:
    """

    f_l = f_shock(x, rho_l, prs_l, gamma) if x > prs_l else f_rare(x, rho_l, prs_l, gamma)
    f_r = f_shock(x, rho_r, prs_r, gamma) if x > prs_r else f_rare(x, rho_r, prs_r, gamma)

    return f_l + f_r + u_r - u_l


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


def calc_u_star(p_star, rho_l, rho_r, u_l, u_r, prs_l, prs_r, gamma):
    """
    Calculates the velocity in the intermediate region and also returns a flag to
    :param p_star: pressure p
    :param rho_l: distant left density
    :param rho_r: distant right density
    :param u_l: distant left velocity
    :param u_r: distant right velocity
    :param prs_l: distant left pressure
    :param prs_r: distant right pressure
    :param gamma: adiabatic index
    :return:
    """

    if p_star > prs_l:
        wave_type_l = 0
        f_l = f_shock(p_star, rho_l, prs_l, gamma)
    else:
        wave_type_l = 1
        f_l = f_rare(p_star, rho_l, prs_l, gamma)

    if p_star > prs_r:
        wave_type_r = 0
        f_r = f_shock(p_star, rho_r, prs_r, gamma)
    else:
        wave_type_r = 1
        f_r = f_rare(p_star, rho_r, prs_r, gamma)

    return wave_type_l, wave_type_r, (u_l + u_r) / 2. + (f_r - f_l) / 2.


def riemann_exact(ql, qr, gamma, grid, time):
    """
    Solves the Riemann problem for an isothermal gas.
    :param ql: left state vector (rho_l, u_l, prs_l).
    :param qr: right state vector (rho_r, u_r, prs_r).
    :param gamma:  adiabatic index
    :param grid: grid on which to evaluate solution. Initial discontinuity is assumed to be at x=0.
    :param time: time at which solution is desired.
    :return: Solution on grid. Same size as grid.
    """

    if time > 0:
        rho_l, rho_r = ql[0], qr[0]
        u_l, u_r = ql[1], qr[1]
        prs_l, prs_r = ql[2], qr[2]
        guess = np.array([1.])

        p_star = sco.newton(p_equation, guess, args=(rho_l, rho_r, u_l, u_r, prs_l, prs_r, gamma))

        wave_type_l, wave_type_r, u_star = calc_u_star(p_star, rho_l, rho_r, u_l, u_r, prs_l, prs_r, gamma)
        print(f'Root found for time {time}: p* = {p_star}, u* = {u_star}')
        shock_str, rare_string = 'shock', 'rarefaction'
        print(f'Left wave is a {shock_str if wave_type_l == 0 else rare_string}.')
        print(f'Right wave is a {shock_str if wave_type_r == 0 else rare_string}.')


        u = sol_u(res, grid, time, u_l, u_r)
        rho = sol_rho(res, grid, u, time, rho_l, rho_r)
        rho = sol_prs(res, grid, u, time, rho_l, rho_r)

    else:
        rho = np.where(grid < 0, ql[0], qr[0])
        u = np.where(grid < 0, ql[1], qr[1])
        prs = np.where(grid < 0, ql[2], qr[2])

    return rho, u, prs
