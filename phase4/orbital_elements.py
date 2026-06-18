import numpy as np

MU = 398600.4418  # km^3/s^2

def rv_to_orbital_elements(r, v):

    r = np.array(r)
    v = np.array(v)

    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    k = np.array([0.0, 0.0, 1.0])

    n = np.cross(k, h)
    n_norm = np.linalg.norm(n)

    e_vec = (
        np.cross(v, h) / MU
        - r / r_norm
    )

    e = np.linalg.norm(e_vec)

    energy = (
        v_norm**2 / 2
        - MU / r_norm
    )

    a = -MU / (2 * energy)

    i = np.degrees(
        np.arccos(
            h[2] / h_norm
        )
    )

    if n_norm > 1e-8:

        raan = np.degrees(
            np.arccos(
                n[0] / n_norm
            )
        )

        if n[1] < 0:
            raan = 360 - raan

    else:
        raan = 0

    if n_norm > 1e-8 and e > 1e-8:

        argp = np.degrees(
            np.arccos(
                np.dot(n, e_vec)
                /
                (n_norm * e)
            )
        )

        if e_vec[2] < 0:
            argp = 360 - argp

    else:
        argp = 0

    if e > 1e-8:

        nu = np.degrees(
            np.arccos(
                np.dot(
                    e_vec,
                    r
                )
                /
                (e * r_norm)
            )
        )

        if np.dot(r, v) < 0:
            nu = 360 - nu

    else:
        nu = 0

    return {
        "semi_major_axis": a,
        "eccentricity": e,
        "inclination": i,
        "raan": raan,
        "arg_perigee": argp,
        "true_anomaly": nu,
        "specific_energy": energy,
        "angular_momentum_mag": h_norm,
    }