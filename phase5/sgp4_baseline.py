from sgp4.api import Satrec
from sgp4.api import jday

def propagate_tle(line1, line2,
                  year, month, day,
                  hour=0, minute=0, second=0):

    sat = Satrec.twoline2rv(
        line1,
        line2
    )

    jd, fr = jday(
        year,
        month,
        day,
        hour,
        minute,
        second
    )

    error, r, v = sat.sgp4(jd, fr)

    return error, r, v