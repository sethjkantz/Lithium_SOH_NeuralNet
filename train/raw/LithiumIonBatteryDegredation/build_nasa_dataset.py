import os
import numpy as np
import pandas as pd

# -- Paths to folder --
META_PATH = "metadata.csv"
DATA_ROOT = "data"   # folder with all csv files

# -- Load metadata --
meta = pd.read_csv(META_PATH)
meta["Capacity"] = pd.to_numeric(meta["Capacity"], errors="coerce")

def summarize_file(fname: str, phase: str):
    """Compute mean current/voltage/temp over active regions."""
    path = os.path.join(DATA_ROOT, fname)
    df = pd.read_csv(path)

    cur = df["Current_measured"]
    volt = df["Voltage_measured"]
    temp = df["Temperature_measured"]

    # Detect active charging/discharging current
    if phase == "discharge":
        mask = cur < -0.01
    elif phase == "charge":
        mask = cur > 0.01
    else:
        mask = cur.abs() > 0.01

    if mask.any():
        c = cur[mask].abs().mean()
        v = volt[mask].mean()
        t = temp[mask].mean()
    else:
        c = cur.abs().mean()
        v = volt.mean()
        t = temp.mean()

    return float(c), float(v), float(t)

rows = []

# -- Loop over each battery and extract cycle summaries --
for bid in meta["battery_id"].unique():
    m_b = meta[meta["battery_id"] == bid].sort_values("uid")

    # all discharge events = cycles
    dis_rows = m_b[m_b["type"] == "discharge"]
    charge_rows = m_b[m_b["type"] == "charge"]

    cycle_number = 1

    for _, drow in dis_rows.iterrows():
        # find nearest charge after this discharge
        later = charge_rows[charge_rows["uid"] > drow["uid"]]

        if not later.empty:
            chfile = later.iloc[0]["filename"]
        else:
            # fallback: last previous charge
            prior = charge_rows[charge_rows["uid"] < drow["uid"]]
            chfile = prior.iloc[-1]["filename"] if not prior.empty else None

        # discharge stats
        disI, disV, disT = summarize_file(drow["filename"], "discharge")

        # charge stats
        if chfile is not None:
            chI, chV, chT = summarize_file(chfile, "charge")
        else:
            chI = chV = chT = np.nan

        rows.append({
            "battery_id": bid,
            "cycle": cycle_number,
            "chI": chI,
            "chV": chV,
            "chT": chT,
            "disI": disI,
            "disV": disV,
            "disT": disT,
            "BCt": drow["Capacity"]  # cycle capacity from NASA metadata
        })

        cycle_number += 1

# -- Build DataFrame --
df = pd.DataFrame(rows)
df = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)

# -- Compute SOH and RUL per battery --
df["SOH"] = np.nan
df["RUL"] = np.nan

for bid in df["battery_id"].unique():
    idx = df["battery_id"] == bid
    sub = df[idx]

    # Use max capacity as 100% reference
    C_ref = sub["BCt"].max()

    df.loc[idx, "SOH"] = sub["BCt"] / C_ref * 100.0

    N = sub["cycle"].max()
    df.loc[idx, "RUL"] = N - sub["cycle"]

# -- reorder to match the other Kaggle dataset --
df = df[[
    "battery_id", "cycle",
    "chI", "chV", "chT",
    "disI", "disV", "disT",
    "BCt", "SOH", "RUL"
]]

out_file = "NASA_Battery_dataset_all.csv"
df.to_csv(out_file, index=False)

print(f"\nDone! Created: {out_file}")
print(df.head())
