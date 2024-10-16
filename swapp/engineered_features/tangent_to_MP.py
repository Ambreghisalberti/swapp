import numpy as np


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


def cartesian_to_tangential(theta, phi, vx, vy, vz):
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


def check(vx, vy, vz, theta, phi):
    vtan1 = V_tan1_to_Shue(theta, phi, vx, vy, vz)
    vtan2 = V_tan2_to_Shue(theta, phi, vx, vy, vz)
    vn = V_normal_to_Shue(theta, phi, vx, vy, vz)
    return abs(np.sqrt(vtan1**2+vtan2**2+vn**2)-np.sqrt(vx**2+vy**2+vz**2))/np.sqrt(vx**2+vy**2+vz**2)