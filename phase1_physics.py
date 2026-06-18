import numpy as np
from skyfield.api import EarthSatellite, load
from poliastro.twobody.orbit import Orbit
from poliastro.bodies import Earth
from astropy import units as u

# -------------------------------------------------
# 1. LOAD SKYFIELD TIME SYSTEM
# -------------------------------------------------
# Skyfield uses its own precise astronomical time system.
# This object lets us create accurate UTC / TT time values.
ts = load.timescale()

# -------------------------------------------------
# 2. LOAD REAL SATELLITE DATA (TLE)
# -------------------------------------------------
# TLE = Two Line Element
# These lines describe the real orbit of an object in space.

tle_sat = (
    "1 25544U 98067A   24007.54791667  .00016717  00000+0  10270-3 0  9993",
    "2 25544  51.6416  19.1944 0004852  88.9935  38.6446 15.50474365429057"
)

tle_debris = (
    "1 40967U 15058B   24007.43246528  .00002182  00000+0  10270-3 0  9994",
    "2 40967  51.6405  21.1045 0004801  87.9935  40.6446 15.50474365429011"
)

# Create satellite objects from TLE 
sat = EarthSatellite(tle_sat[0], tle_sat[1])#one satelite here iss
debris = EarthSatellite(tle_debris[0], tle_debris[1])#debries

# -------------------------------------------------
# 3. PROPAGATE BOTH OBJECTS & FIND TCA
# -------------------------------------------------
# Create a time array:
# From 8 Jan 2026, 00:00 to 06:00
# Every 10 seconds
times = ts.utc(2026, 1, 8, range(0, 6 * 3600, 10))#sequence of precise time of every 10 sec for 6hrs



# For every time instant:
# 1. Get satellite position
# 2. Get debris position
# 3. Compute distance between them\
distances = []
for ti in times:
    r_sat = sat.at(ti).position.km
    r_deb = debris.at(ti).position.km
    distances.append(np.linalg.norm(r_deb - r_sat))#formula of min distance root((x2​−x1​)2+(y2​−y1​)2+(z2​−z1​)2)

distances = np.array(distances)#It calculates the physical separation between the satellite and the debris at each moment in time.

# Find the index where distance is minimum
#This part of the code finds when the satellite and debris come closest to each other and how close they get.
idx = np.argmin(distances)#returns the index of the smallest value in that array.


# Time of Closest Approach (TCA)
tca_time = times[idx]#Uses that index to fetch the corresponding time.

# Minimum separation distance
miss_distance = distances[idx]#Retrieves the smallest distance value.#This is the minimum separation between the satellite and debris (in km).


print("TCA:", tca_time.utc_strftime())
print("Miss distance (km):", miss_distance)

# -------------------------------------------------
# 4. DEFINE MANEUVER TIME (7 DAYS BEFORE TCA)
# -------------------------------------------------
# We plan the avoidance maneuver 7 days before collision.
t_maneuver = ts.utc(
    tca_time.utc.year,
    tca_time.utc.month,
    tca_time.utc.day - 7,
    tca_time.utc.hour,
    tca_time.utc.minute,
    tca_time.utc.second
)#“Schedule the maneuver at the same clock time, but 7 days earlier than the closest-approach moment.”


print("Maneuver time:", t_maneuver.utc_strftime())
#######mechanism upto here##########
# First, compute TCA (when the risk is highest)
# Then, choose a lead time (here, 7 days)
# Plan the orbital correction maneuver sufficiently early to safely alter the trajectory

# -------------------------------------------------
# 5. GET SATELLITE STATE AT MANEUVER TIME
# -------------------------------------------------
# r_m = position at maneuver time
# v_m = velocity at maneuver time
r_m = sat.at(t_maneuver).position.km
v_m = sat.at(t_maneuver).velocity.km_per_s

# -------------------------------------------------
# 6. APPLY SMALL DELTA-V MANEUVER
# -------------------------------------------------
# dv = small change in velocity (0.5 m/s = 0.0005 km/s)
dv = np.array([0.0005, 0, 0])

# New velocity after engine burn
v_m_new = v_m + dv

# Create a new orbit using:
# - Earth gravity
# - Position at maneuver time
# - New velocity after burn
orbit_maneuvered = Orbit.from_vectors(
    Earth,
    r_m * u.km,
    v_m_new * u.km / u.s
)

# -------------------------------------------------
# 7. PROPAGATE AFTER MANEUVER & RECHECK DISTANCES
# -------------------------------------------------
# Create times from maneuver moment to next 14 days
# One sample every 60 seconds
times_after = [
    ts.tt_jd(t_maneuver.tt + i / 86400)
    for i in range(0, 14 * 24 * 3600, 60)
]

distances_after = []

for ti in times_after:
    # Time difference from maneuver
    dt = (ti.tt - t_maneuver.tt) * u.s

    # New satellite position after maneuver
    r_new = orbit_maneuvered.propagate(dt).r.to(u.km).value

    # Debris position
    r_deb = debris.at(ti).position.km

    # Distance between them
    distances_after.append(np.linalg.norm(r_deb - r_new))

distances_after = np.array(distances_after)

# New closest distance after maneuver
new_miss_distance = distances_after.min()

print("New miss distance after maneuver (km):", new_miss_distance)

# -------------------------------------------------
# 8. COLLISION PROBABILITY (INTUITIVE MODEL)
# -------------------------------------------------
# Converts miss distance into a risk value (0 → safe, 1 → dangerous)
def collision_probability(miss_distance_km, sigma_km):
    return np.exp(-(miss_distance_km**2) / (2 * sigma_km**2))

# Two cases:
# - High precision tracking (50 m uncertainty)
# - Low precision tracking (500 m uncertainty)
pc_low_uncertainty = collision_probability(new_miss_distance, 0.05)
pc_high_uncertainty = collision_probability(new_miss_distance, 0.5)

print("Pc (low uncertainty):", pc_low_uncertainty)
print("Pc (high uncertainty):", pc_high_uncertainty)
print("Number of post-maneuver samples:", len(distances_after))
