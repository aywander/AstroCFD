import numpy as np
import scipy.optimize as sco


def f_shock(x, rho, prs, gamma):

    a = 2. / ((gamma + 1.) * rho)
    b = (gamma - 1.) / (gamma + 1.) * prs
    return (x - prs) * np.sqrt(a / (x + b))


def f_rarefaction(x, rho, prs, gamma):

    a = np.sqrt(gamma * prs / rho)

    return (2. * a) / (gamma - 1.) * ((x / prs) ** ((gamma - 1.) / (2. * gamma)) - 1.)


# def p_equation(x, rho_l, rho_r, u_l, u_r, prs_l, prs_r, gamma):
def p_equation(x, q_l, q_r, gamma):
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

    rho_l, u_l, prs_l = q_l[0], q_l[1], q_l[2]
    rho_r, u_r, prs_r = q_r[0], q_r[1], q_r[2]

    f_l = f_shock(x, rho_l, prs_l, gamma) if x > prs_l else f_rarefaction(x, rho_l, prs_l, gamma)
    f_r = f_shock(x, rho_r, prs_r, gamma) if x > prs_r else f_rarefaction(x, rho_r, prs_r, gamma)

    return f_l + f_r + u_r - u_l


def mass_flux(q, q_star):

    rho, u, prs = q[0], q[1], q[2]
    rho_star, u_star, prs_star = q_star[0], q_star[1], q_star[2]

    return -(prs_star - prs) / (u_star - u)


def shock_speed(q, q_star):
    """
    Returns the shock speed given the upstream density, speed and pressure and star values
    :param rho:
    :param u:
    :param prs:
    :param u_star:
    :param prs_star:
    :return:
    """

    rho, u, prs = q[0], q[1], q[2]

    return u - mass_flux(q, q_star) / rho


def rarefaction_speeds(q, q_star, gamma, char_dir):
    """
    :param rho: Left or right density
    :param u: Left or right velocity
    :param prs: Left or right pressure
    :param rho_star: Left or right density in star region
    :param u_star: Speed in star region
    :param prs_star: Pressure in star region
    :param char_dir: -1 or 1 depending on whether the characteristics (rarefaction head and tail) are left or
    right-travelling
    :return:
    """

    rho, u, prs = q[0], q[1], q[2]
    rho_star, u_star, prs_star = q_star[0], q_star[1], q_star[2]

    a = np.sqrt(gamma * prs / rho)
    a_star = np.sqrt(gamma * prs_star / rho_star)

    c1, c2 = u + char_dir * a, u_star + char_dir * a_star
    return min(c1, c2), max(c1, c2)


def identify_regions(grid, time, q_l, q_r, q_star_l, q_star_r, gamma, wave_type_l, wave_type_r):
    """
    Identify regions in the solution. Return a list of regions from left to right.
    :param grid: grid - a 1D array of position values
    :param time: Time at which solution and regions are sought
    :return: List of arrays of the same size as grid with boolean values
             indicating whether the position value on the grid belongs to the corresponding region or not.
             These arrays can be used as slices for other arrays.
    """

    # Create a list of wave-speeds from left to right
    wave_speeds = []
    wave_speeds += [shock_speed(q_l, q_star_l)] if wave_type_l == 0 else rarefaction_speeds(q_l, q_star_l, gamma, -1)
    wave_speeds += [q_star_l[1]]
    wave_speeds += [shock_speed(q_r, q_star_r)] if wave_type_r == 0 else rarefaction_speeds(q_r, q_star_r, gamma, 1)

    # Get consecutive regions from left
    regions = []
    region_prev = np.array([False] * grid.size)
    for ws in wave_speeds:
        region_left_of_wave = grid - ws * time < 0
        regions += [np.logical_and(~region_prev, region_left_of_wave)]
        region_prev = region_left_of_wave

    # Add right-most region
    regions += [~region_prev]
    return regions


def calc_u_star(prs_star, q_l, q_r, gamma):
    """
    Calculates the velocity in the intermediate region and also returns a flag to
    :param prs_star: pressure in star region
    :param q_l: distant left state vector
    :param q_r: distant right state vector
    :param gamma: adiabatic index
    :return:
    """

    rho_l, u_l, prs_l = q_l[0], q_l[1], q_l[2]
    rho_r, u_r, prs_r = q_r[0], q_r[1], q_r[2]

    if prs_star > prs_l:
        wave_type_l = 0
        f_l = f_shock(prs_star, rho_l, prs_l, gamma)
    else:
        wave_type_l = 1
        f_l = f_rarefaction(prs_star, rho_l, prs_l, gamma)

    if prs_star > prs_r:
        wave_type_r = 0
        f_r = f_shock(prs_star, rho_r, prs_r, gamma)
    else:
        wave_type_r = 1
        f_r = f_rarefaction(prs_star, rho_r, prs_r, gamma)

    return wave_type_l, wave_type_r, (u_l + u_r) / 2. + (f_r - f_l) / 2.


def calc_rho_star(u_star, prs_star, q_l, q_r, gamma, wave_type_l, wave_type_r):

    if wave_type_l == 0:
        q_star = [0., u_star, prs_star]
        mf = mass_flux(q_l, q_star)
        vs = shock_speed(q_l, q_star)
        rho_star_l = mf / (u_star - vs)

    else:
        rho, u, prs = q_l[0], q_l[1], q_l[2]
        rho_star_l = rho * (prs_star / prs) ** (1. / gamma)

    if wave_type_r == 0:
        q_star = [0., u_star, prs_star]
        mf = mass_flux(q_r, q_star)
        vs = shock_speed(q_r, q_star)
        rho_star_r = mf / (u_star - vs)

    else:
        rho, u, prs = q_r[0], q_r[1], q_r[2]
        rho_star_r = rho * (prs_star / prs) ** (1. / gamma)

    return rho_star_l, rho_star_r


def construct_solution(grid, time, q_l, q_r, q_star_l, q_star_r, gamma, wave_type_l, wave_type_r):

    regions = identify_regions(grid, time, q_l, q_r, q_star_l, q_star_r, gamma, wave_type_l, wave_type_r)

    rho_l, u_l, prs_l = q_l[0], q_l[1], q_l[2]
    rho_r, u_r, prs_r = q_r[0], q_r[1], q_r[2]
    rho_star_l, u_star_l, prs_star_l = q_star_l[0], q_star_l[1], q_star_l[2]
    rho_star_r, u_star_r, prs_star_r = q_star_r[0], q_star_r[1], q_star_r[2]

    rho = np.zeros_like(grid)
    u = np.zeros_like(grid)
    prs = np.zeros_like(grid)

    reg = regions[0]
    rho[reg] = rho_l
    u[reg] = u_l
    prs[reg] = prs_l

    reg = regions[-1]
    rho[reg] = rho_r
    u[reg] = u_r
    prs[reg] = prs_r

    if wave_type_l == 0:
        reg = regions[1]
        rho[reg] = q_star_l[0]
        u[reg] = q_star_l[1]
        prs[reg] = q_star_l[2]

    else:
        reg = regions[1]
        u[reg] = u_l + (grid[reg] - grid[reg][0]) / (grid[reg][-1] - grid[reg][0]) * (u_star_l - u_l)
        a_l = np.sqrt(gamma * prs_l / rho_l)
        j = u_l + 2. * a_l / (gamma - 1.)
        a = (j - u[reg]) * (gamma - 1.) / 2.
        rho[reg] = rho_l * (a / a_l) ** (2. / (gamma - 1.))
        prs[reg] = prs_l * (rho[reg] / rho_l) ** gamma

        reg = regions[2]
        rho[reg] = q_star_l[0]
        u[reg] = q_star_l[1]
        prs[reg] = q_star_l[2]

    if wave_type_r == 0:
        reg = regions[-2]
        rho[reg] = q_star_r[0]
        u[reg] = q_star_r[1]
        prs[reg] = q_star_r[2]

    else:
        reg = regions[-2]
        u[reg] = u_star_r + (grid[reg] - grid[reg][0]) / (grid[reg][-1] - grid[reg][0]) * (u_r - u_star_r)
        a_r = np.sqrt(gamma * prs_r / rho_r)
        j = u_r - 2. * a_r / (gamma - 1.)
        a = -(j - u[reg]) * (gamma - 1.) / 2.
        rho[reg] = rho_r * (a / a_r) ** (2. / (gamma - 1.))
        prs[reg] = prs_r * (rho[reg] / rho_r) ** gamma

        reg = regions[-3]
        rho[reg] = q_star_r[0]
        u[reg] = q_star_r[1]
        prs[reg] = q_star_r[2]

    return rho, u, prs


def riemann_exact(q_l, q_r, gamma, grid, time):
    """
    Solves the Riemann problem for an isothermal gas.
    :param q_l: left state vector (rho_l, u_l, prs_l).
    :param q_r: right state vector (rho_r, u_r, prs_r).
    :param gamma:  adiabatic index
    :param grid: grid on which to evaluate solution. Initial discontinuity is assumed to be at x=0.
    :param time: time at which solution is desired.
    :return: Solution on grid. Same size as grid.
    """

    rho_l, u_l, prs_l = q_l[0], q_l[1], q_l[2]
    rho_r, u_r, prs_r = q_r[0], q_r[1], q_r[2]

    if time > 0:

        guess = (prs_r + prs_l) / 2.
        prs_star = sco.newton(p_equation, guess, args=(q_l, q_r, gamma))

        wave_type_l, wave_type_r, u_star = calc_u_star(prs_star, q_l, q_r, gamma)

        rho_star_l, rho_star_r = calc_rho_star(u_star, prs_star, q_l, q_r, gamma, wave_type_l, wave_type_r)

        q_star_l = [rho_star_l, u_star, prs_star]
        q_star_r = [rho_star_r, u_star, prs_star]

        print(f'Root found:')
        print(f'p* = {prs_star}, u* = {u_star}')
        print(f'rho_*l = {rho_star_l}, rho_r = {rho_star_r}')
        shock_str, rare_str= 'shock', 'rarefaction'
        print(f'Left wave is a {shock_str if wave_type_l == 0 else rare_str}.')
        print(f'Right wave is a {shock_str if wave_type_r == 0 else rare_str}.')

        rho, u, prs = construct_solution(grid, time, q_l, q_r, q_star_l, q_star_r, gamma, wave_type_l, wave_type_r)

    else:
        rho = np.where(grid < 0, q_l[0], q_r[0])
        u = np.where(grid < 0, q_l[1], q_r[1])
        prs = np.where(grid < 0, q_l[2], q_r[2])

    return rho, u, prs


#%%
