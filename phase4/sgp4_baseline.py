from sgp4.api import Satrec
from sgp4.api import jday

from datetime import datetime
from datetime import timezone


def tle_epoch_to_datetime(tle1):

    epoch_str = tle1[18:32].strip()

    yy = int(epoch_str[:2])

    if yy < 57:
        year = 2000 + yy
    else:
        year = 1900 + yy

    day = float(epoch_str[2:])

    return datetime(
        year,
        1,
        1,
        tzinfo=timezone.utc
    ) + timedelta(days=day - 1)


def propagate_tle(
    tle1,
    tle2,
    timestamp
):

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

    error, r, v = sat.sgp4(
        jd,
        fr
    )

    if error != 0:
        return None

    return [
        r[0],
        r[1],
        r[2],
        v[0],
        v[1],
        v[2]
    ]