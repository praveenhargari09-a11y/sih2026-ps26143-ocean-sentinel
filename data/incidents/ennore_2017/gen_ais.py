"""
Regenerate the Ennore 2017 AIS traffic data with geographically correct
vessel tracks that approach Kamarajar Port (Ennore) from the sea (Bay of Bengal)
rather than crossing over land.

Key facts:
- Kamarajar Port (Ennore) is at approximately 13.228, 80.363
- The coastline at Ennore runs roughly N-S, coast at about lon 80.30
- Ships approach from the east/southeast (open Bay of Bengal)
- MT Dawn Kanchipuram: Oil Products Tanker, MMSI 419000988
  → Approaches from the SE, arrives at port, collision at ~22:30Z
  → AIS gap from 22:30 to 02:40 (after collision)
  → Reappears near port at low speed
- MT BW Maple: LPG Tanker, MMSI 235101303
  → Approaches from the NE (coming from Visakhapatnam direction)
  → Arrives at collision area, slows down after collision
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ais():
    start_time = datetime(2017, 1, 27, 12, 0, 0)
    collision_time = datetime(2017, 1, 27, 22, 30, 0)
    end_time = datetime(2017, 1, 28, 4, 30, 0)
    delta = timedelta(minutes=10)

    # Port / collision location
    port_lat = 13.228166
    port_lon = 80.363333

    records = []

    # ── MT Dawn Kanchipuram (MMSI 419000988) ────────────────────────────────
    # Approaches from the SOUTHEAST through the Bay of Bengal
    # Start position: open sea SE of Chennai (well offshore)
    dawn_start_lat = 12.60
    dawn_start_lon = 81.20    # ~90km offshore in Bay of Bengal

    # Track: sail NW toward Ennore Port (approach from sea)
    # Pre-collision phase: 12:00 to 22:20
    t = start_time
    while t <= datetime(2017, 1, 27, 22, 20, 0):
        # Fraction of approach completed
        total_min = (datetime(2017, 1, 27, 22, 20, 0) - start_time).total_seconds() / 60.0
        elapsed_min = (t - start_time).total_seconds() / 60.0
        frac = elapsed_min / total_min

        # Smooth approach curve — slight arc to simulate realistic sea route
        # Use a quadratic easing to slow as vessel nears port
        lat = dawn_start_lat + (port_lat - dawn_start_lat) * frac
        # Longitude: start far east in Bay of Bengal, curve toward coast
        # Slight eastward bulge in the first half to keep route over water
        arc_offset = 0.15 * np.sin(frac * np.pi)  # bulge east mid-route
        lon = dawn_start_lon + (port_lon - dawn_start_lon) * frac + arc_offset

        # Heading: NW approach (~315-340°)
        heading = 330.0 - 15.0 * frac  # gradually straightens toward port
        cog = heading

        records.append({
            'mmsi': '419000988',
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'base_date_time': t.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'sog': 10.0,
            'cog': round(cog, 1),
            'heading': round(heading, 1),
            'vessel_name': 'Dawn Kanchipuram',
            'vessel_type': 'Oil Products Tanker',
            'imo': '9116917',
            'status': '0'
        })
        t += delta

    # AIS GAP: 22:30 to 02:30 (collision + response period, 4 hours)
    # No records during this window

    # Post-collision: stationary near port (02:40 to 04:30)
    t = datetime(2017, 1, 28, 2, 40, 0)
    drift_idx = 0
    while t <= end_time:
        # Vessel drifts very slowly near collision site
        lat = port_lat + 0.002 * drift_idx  # tiny northward drift
        lon = port_lon + 0.001  # just offshore of port

        records.append({
            'mmsi': '419000988',
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'base_date_time': t.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'sog': 0.5,
            'cog': 0.0,
            'heading': 0.0,
            'vessel_name': 'Dawn Kanchipuram',
            'vessel_type': 'Oil Products Tanker',
            'imo': '9116917',
            'status': '0'
        })
        t += delta
        drift_idx += 1

    # ── MT BW Maple (MMSI 235101303) ────────────────────────────────────────
    # Approaches from the NORTH/NORTHEAST (from Visakhapatnam direction)
    # along the coast, staying offshore
    bw_start_lat = 14.80
    bw_start_lon = 80.65    # offshore, NE of Ennore

    # Pre-collision: 12:00 to 22:30 (southward approach along coast)
    t = start_time
    while t <= collision_time:
        total_min = (collision_time - start_time).total_seconds() / 60.0
        elapsed_min = (t - start_time).total_seconds() / 60.0
        frac = elapsed_min / total_min

        lat = bw_start_lat + (port_lat - bw_start_lat) * frac
        # Keep well offshore throughout — slight eastward arc
        arc_offset = 0.10 * np.sin(frac * np.pi)
        lon = bw_start_lon + (port_lon - bw_start_lon) * frac + arc_offset

        heading = 195.0 + 5.0 * frac
        cog = heading

        records.append({
            'mmsi': '235101303',
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'base_date_time': t.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'sog': 14.0,
            'cog': round(cog, 1),
            'heading': round(heading, 1),
            'vessel_name': 'BW Maple',
            'vessel_type': 'LPG Tanker',
            'imo': '9320752',
            'status': '0'
        })
        t += delta

    # Post-collision: BW Maple slows and drifts south (22:40 to 04:30)
    t = datetime(2017, 1, 27, 22, 40, 0)
    drift_idx = 0
    while t <= end_time:
        lat = port_lat - 0.003 * drift_idx  # slow southward drift
        lon = port_lon - 0.001 * drift_idx   # slight westward drift (still offshore)

        records.append({
            'mmsi': '235101303',
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'base_date_time': t.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'sog': 3.0,
            'cog': 195.0,
            'heading': 195.0,
            'vessel_name': 'BW Maple',
            'vessel_type': 'LPG Tanker',
            'imo': '9320752',
            'status': '0'
        })
        t += delta
        drift_idx += 1

    df = pd.DataFrame(records)
    # Sort by time, then by MMSI for clean output
    df = df.sort_values(['base_date_time', 'mmsi']).reset_index(drop=True)
    output_path = r'c:\Users\prave\Desktop\Oil Spill\data\incidents\ennore_2017\ais_traffic.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} AIS records to {output_path}")

    # Validate: print Dawn Kanchipuram summary
    dawn = df[df['mmsi'] == '419000988']
    bw = df[df['mmsi'] == '235101303']
    print(f"\nDawn Kanchipuram: {len(dawn)} points")
    print(f"  Start: lat={dawn.iloc[0]['latitude']}, lon={dawn.iloc[0]['longitude']} (should be offshore)")
    print(f"  End:   lat={dawn.iloc[-1]['latitude']}, lon={dawn.iloc[-1]['longitude']} (near port)")
    print(f"  All lon >= 80.30? {(dawn['longitude'] >= 80.30).all()}")
    print(f"\nBW Maple: {len(bw)} points")
    print(f"  Start: lat={bw.iloc[0]['latitude']}, lon={bw.iloc[0]['longitude']} (should be offshore)")
    print(f"  End:   lat={bw.iloc[-1]['latitude']}, lon={bw.iloc[-1]['longitude']}")
    print(f"  All lon >= 80.30? {(bw['longitude'] >= 80.30).all()}")

if __name__ == '__main__':
    generate_ais()
