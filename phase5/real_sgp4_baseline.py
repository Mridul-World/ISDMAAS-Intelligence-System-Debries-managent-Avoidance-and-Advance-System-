import pandas as pd
import numpy as np

from sgp4.api import Satrec
from sgp4.api import jday

def propagate_tle(tle1, tle2, timestamp):

    sat = Satrec.twoline2rv(
        tle1,
        tle2
    )

    jd, fr = jday(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second
    )

    e, r, v = sat.sgp4(jd, fr)

    if e != 0:
        return None

    return np.array([
        r[0],
        r[1],
        r[2],
        v[0],
        v[1],
        v[2]
    ])