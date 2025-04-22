import numpy as np
from spok.models.planetary import mp_shue1998_tangents, mp_shue1998_normal


def get_shue_params(**kwargs):
    bz = kwargs.get('Bz', -0.001)
    pd = kwargs.get('Pd', 2.056)

    r0 = (11.4 + 0.13 * bz) * pd ** (-1 / 6.6)
    alpha = (0.58 - 0.010 * bz) * (1 + 0.010 * pd)
    return r0, alpha


def r_shue(theta, **kwargs):
    r0, alpha = get_shue_params(**kwargs)
    return r0 * (2 / (1 + np.cos(theta))) ** alpha


def V_normal_to_Shue(theta, phi, vx, vy, vz, **kwargs):
    r0, alpha = get_shue_params(**kwargs)
    const = 2 * alpha * np.sin(theta) / (1 + np.cos(theta))
    denominateur = np.sqrt(1 + const ** 2)
    numerateur = (vx * np.cos(theta) + vy * np.sin(theta) * np.sin(phi) + 
                  vz * np.sin(theta) * np.cos(phi))
    numerateur -= const * (-vx * np.sin(theta) + vy * np.cos(theta) * np.sin(phi) + 
                           vz * np.cos(theta) * np.cos(phi))
    return numerateur / denominateur


def V_tan1_to_Shue(theta, phi, vx, vy, vz, **kwargs):
    r0, alpha = get_shue_params(**kwargs)
    const = 2 * alpha * np.sin(theta) / (1 + np.cos(theta))
    denominateur = np.sqrt(1 + const ** 2)
    numerateur = (-vx * np.sin(theta) + vy * np.cos(theta) * np.sin(phi) + vz * np.cos(theta) * np.cos(phi))
    numerateur += const * (vx * np.cos(theta) + vy * np.sin(theta) * np.sin(phi) + vz * np.sin(theta) * np.cos(phi))
    return numerateur / denominateur


def V_tan2_to_Shue(theta, phi, vx, vy, vz, **kwargs):
    return vy * np.cos(phi) - vz * np.sin(phi)


def cartesian_to_tangential0(theta, phi, vx, vy, vz):
    vtan1 = V_tan1_to_Shue(theta, phi, vx, vy, vz)
    vtan2 = V_tan2_to_Shue(theta, phi, vx, vy, vz)
    vn = V_normal_to_Shue(theta, phi, vx, vy, vz)
    '''if isinstance(vtan1, np.ndarray):
        assert ((vtan1**2 + vtan2**2 + vn**2)!=(vx**2 + vy**2 + vz**2)).sum_value()==0
    else:
        assert (vtan1**2 + vtan2**2 + vn**2)==(vx**2 + vy**2 + vz**2)
    '''
    return vtan1, vtan2, vn


def check_tan1_unitary(theta, phi, **kwargs):
    r0, alpha = get_shue_params(**kwargs)
    const = 2*alpha*np.sin(theta)/(1+np.cos(theta))
    denominateur = np.sqrt(1 + const**2)
    sum_value = (const*np.cos(theta) - np.sin(theta))**2
    sum_value += (const*np.sin(theta)*np.sin(phi) + np.cos(theta)*np.sin(phi))**2
    sum_value += (const*np.sin(theta)*np.cos(phi) + np.cos(theta)*np.cos(phi))**2
    return sum_value/denominateur**2


def check_normal_unitary(theta, phi, **kwargs):
    r0, alpha = get_shue_params(**kwargs)
    const = 2*alpha*np.sin(theta)/(1+np.cos(theta))
    denominateur = np.sqrt(1 + const**2)
    sum_value = (const*np.sin(theta) + np.cos(theta))**2
    sum_value += (-const*np.cos(theta)*np.sin(phi) + np.sin(theta)*np.sin(phi))**2
    sum_value += (-const*np.cos(theta)*np.cos(phi) + np.sin(theta)*np.cos(phi))**2
    return sum_value/denominateur**2


def check_x_unitary(theta, phi, **kwargs):
    r0, alpha = get_shue_params(**kwargs)
    const = 2*alpha*np.sin(theta)/(1+np.cos(theta))
    denominateur = np.sqrt(1 + const**2)
    sum_value = (const*np.cos(theta) - np.sin(theta))**2/denominateur**2  # ex.etan1
    sum_value += 0  # ex.etan2
    sum_value += (np.cos(theta)+const*np.sin(theta))**2/denominateur**2  # ex.en
    return sum_value


def check_y_unitary(theta, phi, **kwargs):
    r0, alpha = get_shue_params(**kwargs)
    const = 2 * alpha * np.sin(theta) / (1 + np.cos(theta))
    denominateur = np.sqrt(1 + const ** 2)

    sum_value = ((const * np.sin(theta) * np.sin(phi) + np.cos(theta) * np.sin(phi)) ** 2 / denominateur ** 2)
    # ey.etan1
    sum_value += np.cos(phi) ** 2  # ey.etan2
    sum_value += ((np.sin(theta) * np.sin(phi) - const * np.cos(theta) * np.sin(phi)) ** 2 / denominateur ** 2)
    # ey.en
    return sum_value


def check_z_unitary(theta, phi, **kwargs):
    r0, alpha = get_shue_params(**kwargs)
    const = 2 * alpha * np.sin(theta) / (1 + np.cos(theta))
    denominateur = np.sqrt(1 + const ** 2)

    sum_value = ((const * np.sin(theta) * np.cos(phi) + np.cos(theta) * np.cos(phi)) ** 2 / denominateur ** 2)
    # ez.etan1
    sum_value += np.sin(phi) ** 2  # ez.etan2
    sum_value += ((np.sin(theta) * np.cos(phi) - const * np.cos(theta) * np.cos(phi)) ** 2 / denominateur ** 2)
    # ez.en
    return sum_value


def check_cartesian_to_tangential(vx, vy, vz, theta, phi):
    vtan1 = V_tan1_to_Shue(theta, phi, vx, vy, vz)
    vtan2 = V_tan2_to_Shue(theta, phi, vx, vy, vz)
    vn = V_normal_to_Shue(theta, phi, vx, vy, vz)
    return abs(np.sqrt(vtan1**2+vtan2**2+vn**2)-np.sqrt(vx**2+vy**2+vz**2))/np.sqrt(vx**2+vy**2+vz**2)


def Vx_from_tangential(theta, phi, vtan1, vtan2, vn, **kwargs):
    r0, alpha = get_shue_params(**kwargs)
    const = 2 * alpha * np.sin(theta) / (1 + np.cos(theta))
    denominateur = np.sqrt(1 + const ** 2)
    a = (const*np.cos(theta)-np.sin(theta))
    b = np.sin(phi)*(np.cos(theta)+const*np.sin(theta))
    c = np.cos(phi)*(np.cos(theta)+const*np.sin(theta))
    return (a*vtan1 + b*vtan2 + c*vn)/denominateur


def Vy_from_tangential(theta, phi, vtan1, vtan2, vn, **kwargs):
    return np.cos(phi)*vtan2 - np.sin(phi)*vn


def Vz_from_tangential(theta, phi, vtan1, vtan2, vn, **kwargs):
    r0, alpha = get_shue_params(**kwargs)
    const = 2 * alpha * np.sin(theta) / (1 + np.cos(theta))
    denominateur = np.sqrt(1 + const ** 2)
    a = (const*np.sin(theta)+np.cos(theta))
    b = np.sin(phi)*(np.sin(theta)-const*np.cos(theta))
    c = np.cos(phi)*(np.sin(theta)-const*np.cos(theta))
    return (a*vtan1 + b*vtan2 + c*vn)/denominateur


def check_tangential_to_cartesian(vtan1, vtan2, vn, theta, phi):
    vx = Vx_from_tangential(theta, phi, vtan1, vtan2, vn)
    vy = Vy_from_tangential(theta, phi, vtan1, vtan2, vn)
    vz = Vz_from_tangential(theta, phi, vtan1, vtan2, vn)
    return abs(np.sqrt(vx**2+vy**2+vz**2)-np.sqrt(vtan1**2+vtan2**2+vn**2))/np.sqrt(vtan1**2+vtan2**2+vn**2)

def check_transformations(vx, vy, vz, theta, phi):
    vtan1 = V_tan1_to_Shue(theta, phi, vx, vy, vz)
    vtan2 = V_tan2_to_Shue(theta, phi, vx, vy, vz)
    vn = V_normal_to_Shue(theta, phi, vx, vy, vz)

    vx_bis = Vx_from_tangential(theta, phi, vtan1, vtan2, vn)
    vy_bis = Vy_from_tangential(theta, phi, vtan1, vtan2, vn)
    vz_bis = Vz_from_tangential(theta, phi, vtan1, vtan2, vn)

    return abs(np.sqrt(vx_bis ** 2 + vy_bis ** 2 + vz_bis ** 2) - np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)) / np.sqrt(
        vx_bis ** 2 + vy_bis ** 2 + vz_bis ** 2)


# This is from Bayane's work, and it gives results that are almost linearly linked to mine, but that are different,
# so I take hers because her calculations are simpler, she's more likely to be right
def cartesian_to_tangential0(theta, phi, valx, valy, valz):
    x_normal, y_normal, z_normal = mp_shue1998_normal(theta, phi)
    [x_tan1, y_tan1, z_tan1], [x_tan2, y_tan2, z_tan2] = mp_shue1998_tangents(theta, phi)

    valn = x_normal * valx + y_normal * valy + z_normal * valz
    valtan1 = x_tan1 * valx + y_tan1 * valy + z_tan1 * valz
    valtan2 = x_tan2 * valx + y_tan2 * valy + z_tan2 * valz

    return valtan1, valtan2, valn