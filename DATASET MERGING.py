import pandas as pd
import numpy as np
import re

# =========================================================
# SEASON DEFINITIONS
# =========================================================
season_oni = {
    'Kharif': ['MJJ', 'JJA', 'JAS', 'ASO'],
    'Rabi':   ['OND', 'NDJ', 'DJF', 'JFM'],
    'Summer': ['MAM', 'AMJ'],
    'Winter': ['DJF', 'JFM'],
}

def tag_season(month):
    if month in [6, 7, 8, 9, 10]: return 'Kharif'
    if month in [11, 12, 1, 2, 3]: return 'Rabi'
    if month in [4, 5]:           return 'Summer'
    return 'Winter'

# =========================================================
# LOAD
# =========================================================
print("Loading files...")
yield_df = pd.read_csv("rice_state_season_yield_1997_2020.csv")
oni_df   = pd.read_csv("Clean_ENSO_Data.csv")
irrig_df = pd.read_csv("rice_irrigation_ratio_corrected.csv")
rain_df  = pd.read_csv("statewise_monthly_rainfall_1997_2024_cleaned.csv")
temp_df  = pd.read_csv("statewise_monthly_temperature_1998_2024.csv")

# 1. YIELD
# =========================================================
print("\n Processing Yield ")
yield_df.columns = [c.strip() for c in yield_df.columns]
yield_df["State"]  = yield_df["State"].astype(str).str.strip().str.title()
yield_df["Season"] = yield_df["Season"].astype(str).str.strip().str.title()

# Parse '1997-98' / '2000 - 2001' → 1997 / 2000 (start-year convention)
yield_df["Year"] = (
    yield_df["Year"].astype(str).str.extract(r"(\d{4})")[0]
)
yield_df = yield_df.dropna(subset=["Year"])
yield_df["Year"] = yield_df["Year"].astype(int)

yield_df = yield_df.dropna(subset=["State", "Season", "Yield_tonne_per_ha"])
yield_df["Yield_tonne_per_ha"] = pd.to_numeric(
    yield_df["Yield_tonne_per_ha"], errors="coerce"
)
yield_df = yield_df.dropna(subset=["Yield_tonne_per_ha"])


yield_clean = yield_df.rename(
    columns={"Yield_tonne_per_ha": "Rice_Yield"}
)[["State", "Year", "Season", "Rice_Yield"]].copy()


yield_clean = yield_clean[yield_clean["Season"] != "Whole Year"].copy()
yield_clean["Season"] = yield_clean["Season"].replace(
    {"Winter": "Rabi", "Autumn": "Kharif"}
)

yield_clean = (
    yield_clean.groupby(["State", "Year", "Season"], as_index=False)
               ["Rice_Yield"].mean()
)

print("Yield shape :", yield_clean.shape)
print("Seasons     :", sorted(yield_clean["Season"].unique()))
print("States      :", yield_clean["State"].nunique())
print("Years       :", yield_clean["Year"].min(), "–", yield_clean["Year"].max())


# 2. IRRIGATION
# =========================================================
print("\n Processing Irrigation")
irrig_df.columns = [c.strip() for c in irrig_df.columns]
irrig_df = irrig_df.rename(columns={"Crop_Year": "Year"})
irrig_df["Year"] = (
    irrig_df["Year"].astype(str).str.extract(r"(\d{4})")[0]
)
irrig_df = irrig_df.dropna(subset=["Year"])
irrig_df["Year"] = irrig_df["Year"].astype(int)

irrig_df["State"]  = irrig_df["State"].astype(str).str.strip().str.title()
irrig_df["Season"] = irrig_df["Season"].astype(str).str.strip().str.title()

needs_fix = (
    irrig_df["Irrigation_Ratio_pct"].isna()
    & (irrig_df["Total_Rice_Area"] > 0)
)
irrig_df.loc[needs_fix, "Irrigation_Ratio_pct"] = (
    irrig_df.loc[needs_fix, "Irrigated_Rice_Area"]
    / irrig_df.loc[needs_fix, "Total_Rice_Area"]
) * 100


irrig_df.loc[irrig_df["Irrigation_Ratio_pct"] < 0,   "Irrigation_Ratio_pct"] = np.nan
irrig_df.loc[irrig_df["Irrigation_Ratio_pct"] > 100, "Irrigation_Ratio_pct"] = 100

irrig_clean = (
    irrig_df[["State", "Year", "Season", "Irrigation_Ratio_pct"]]
    .dropna(subset=["Irrigation_Ratio_pct"])
    .rename(columns={"Irrigation_Ratio_pct": "Irrigation_Pct"})
    .copy()
)


irrig_clean = irrig_clean[irrig_clean["Season"] != "Total"].copy()
irrig_clean["Season"] = irrig_clean["Season"].replace(
    {"Winter": "Rabi", "Autumn": "Kharif"}
)


irrig_clean = (
    irrig_clean.groupby(["State", "Year", "Season"], as_index=False)
               ["Irrigation_Pct"].mean()
)

print("Irrig shape :", irrig_clean.shape)
print("Seasons     :", sorted(irrig_clean["Season"].unique()))


# 3. RAINFALL
# =========================================================
print("\n Processing Rainfall ")
rain_df.columns = [c.strip() for c in rain_df.columns]
# Split '2023-01' → Year + Month_Num
rain_df[["Year", "Month_Num"]] = (
    rain_df["Month"].astype(str).str.extract(r"(\d{4})-(\d{1,2})")
)

rain_df.groupby(["State", "Year", "Month_Num"]).size().describe()
rain_df["Year"]      = pd.to_numeric(rain_df["Year"],      errors="coerce")
rain_df["Month_Num"] = pd.to_numeric(rain_df["Month_Num"], errors="coerce")
rain_df = rain_df.dropna(subset=["Year", "Month_Num"])
rain_df["Year"]      = rain_df["Year"].astype(int)
rain_df["Month_Num"] = rain_df["Month_Num"].astype(int)
rain_df["State"]     = rain_df["State"].astype(str).str.strip().str.title()

rain_df["Season"] = rain_df["Month_Num"].apply(tag_season)

rain_df["Season_Year"] = np.where(
    (rain_df["Season"] == "Rabi") & (rain_df["Month_Num"].isin([11, 12])),
    rain_df["Year"] + 1,
    rain_df["Year"]
)

rain_clean = (
    rain_df.groupby(["State", "Season_Year", "Season"], as_index=False)
           ["Monthly_Rainfall_mm"].sum(min_count=1)
           .rename(columns={"Season_Year": "Year",
                            "Monthly_Rainfall_mm": "Season_Rainfall_mm"})
)
rain_clean = rain_clean.dropna(subset=["Season_Rainfall_mm"])

print("Rain shape  :", rain_clean.shape)
print("Seasons     :", sorted(rain_clean["Season"].unique()))


# 4. TEMPERATURE
# =========================================================
print("\n Processing Temperature ")
temp_df.columns = [c.strip() for c in temp_df.columns]

temp_df[["Year", "Month_Num"]] = (
    temp_df["Month"].astype(str).str.extract(r"(\d{4})-(\d{1,2})")
)
temp_df["Year"]      = pd.to_numeric(temp_df["Year"],      errors="coerce")
temp_df["Month_Num"] = pd.to_numeric(temp_df["Month_Num"], errors="coerce")
temp_df = temp_df.dropna(subset=["Year", "Month_Num"])
temp_df["Year"]      = temp_df["Year"].astype(int)
temp_df["Month_Num"] = temp_df["Month_Num"].astype(int)
temp_df["State"]     = temp_df["State"].astype(str).str.strip().str.title()

temp_df["Season"] = temp_df["Month_Num"].apply(tag_season)
temp_df["Season_Year"] = np.where(
    (temp_df["Season"] == "Rabi") & (temp_df["Month_Num"].isin([11, 12])),
    temp_df["Year"] + 1,
    temp_df["Year"]
)

temp_clean = (
    temp_df.groupby(["State", "Season_Year", "Season"], as_index=False)
           .agg(Season_Tmax_C=("Mean_Tmax_C", "mean"),
                Season_Tmin_C=("Mean_Tmin_C", "mean"))
           .rename(columns={"Season_Year": "Year"})
)
temp_clean["Season_Temp_C"] = (
    temp_clean["Season_Tmax_C"] + temp_clean["Season_Tmin_C"]
) / 2

print("Temp shape  :", temp_clean.shape)
print("Seasons     :", sorted(temp_clean["Season"].unique()))


# 5. ONI per (Year, Season)
# =========================================================

print("\n Processing ONI ")


if "SEAS" in oni_df.columns:
    # Already long: Year, SEAS, ANOM
    oni_long = oni_df.rename(columns={"SEAS": "window", "ANOM": "oni"})[
        ["Year", "window", "oni"]
    ].copy()
else:
    # Wide: Year, DJF, JFM, ...
    oni_long = oni_df.melt(
        id_vars=["Year"], var_name="window", value_name="oni"
    )
    oni_long["window"] = oni_long["window"].astype(str).str.strip().str.upper()

oni_long["Year"] = pd.to_numeric(oni_long["Year"], errors="coerce")
oni_long["oni"]  = pd.to_numeric(oni_long["oni"], errors="coerce")
oni_long = oni_long.dropna(subset=["Year", "oni"])
oni_long["Year"] = oni_long["Year"].astype(int)


oni_long["Season"] = oni_long["window"].apply(
    lambda w: next((s for s, ws in season_oni.items() if w in ws), None)
)

print("ONI windows seen   :", sorted(oni_long["window"].unique())[:15])
print("ONI unmapped count :", oni_long["Season"].isna().sum(),
      "/", len(oni_long))

oni_clean = (
    oni_long.dropna(subset=["Season"])
    .groupby(["Year", "Season"], as_index=False)["oni"].mean()
    .rename(columns={"oni": "Season_ONI"})
)
print("ONI clean shape    :", oni_clean.shape)
print("ONI seasons        :", sorted(oni_clean["Season"].unique()))
print("ONI year range     :", oni_clean["Year"].min(), "–", oni_clean["Year"].max())
print("ONI value range    :", round(oni_clean["Season_ONI"].min(), 2),
                          "–", round(oni_clean["Season_ONI"].max(), 2))


# 6. STATE NAME ALIGNMENT
# =========================================================
print("\n State name alignment")
state_map = {
    "ORISSA": "ODISHA",
    "JAMMU & KASHMIR": "JAMMU AND KASHMIR",
    "JAMMU AND KASHMIR ": "JAMMU AND KASHMIR",
    "A & N ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
    "ANDAMAN & NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
    "PONDICHERRY": "PUDUCHERRY",
    "UTTARANCHAL": "UTTARAKHAND",
    "DADRA & NAGAR HAVELI": "DADRA AND NAGAR HAVELI",
    "DAMAN & DIU": "DAMAN AND DIU",
    "TELANGANA ": "TELANGANA",
}

def align_state(df):
    df["State"] = (
        df["State"].astype(str).str.strip()
        .replace(state_map)
    )
    return df

yield_clean = align_state(yield_clean)
irrig_clean = align_state(irrig_clean)
rain_clean  = align_state(rain_clean)
temp_clean  = align_state(temp_clean)


def suset(df): return set(df["State"].astype(str).str.upper())
y = suset(yield_clean); i = suset(irrig_clean)
r = suset(rain_clean);  t = suset(temp_clean)

print("Yield ∩ Irrig :", len(y & i))
print("Yield ∩ Rain  :", len(y & r))
print("Yield ∩ Temp  :", len(y & t))
print("Yield not in Rain :", sorted(y - r)[:10])
print("Yield not in Temp :", sorted(y - t)[:10])


# 7. MERGE (left-join from yield to preserve all seasons)
# =========================================================
print("\n--- Merging ---")
final_df = (
    yield_clean
    .merge(irrig_clean, on=["State", "Year", "Season"], how="left")
    .merge(rain_clean,  on=["State", "Year", "Season"], how="left")
    .merge(temp_clean,  on=["State", "Year", "Season"], how="left")
    .merge(oni_clean,   on=["Year", "Season"],          how="left")
)

print("After merge rows:", len(final_df))


final_df["Irrigation_Imputed"] = final_df["Irrigation_Pct"].isna().astype(int)
final_df["Irrigation_Pct"] = (
    final_df.groupby("State")["Irrigation_Pct"]
            .transform(lambda x: x.fillna(x.mean()))
)
final_df["Irrigation_Imputed"] = final_df["Irrigation_Imputed"].fillna(0).astype(int)
final_df["Irrigation_Pct"] = (
    final_df.groupby("State")["Irrigation_Pct"]
            .transform(lambda x: x.fillna(x.mean()))
)
print(final_df.nsmallest(10, "Season_Temp_C")[
    ["State", "Year", "Season", "Season_Temp_C"]
])
print(final_df[["Season_Tmax_C", "Season_Tmin_C"]].describe())
print((final_df["Season_Tmin_C"] > final_df["Season_Tmax_C"]).sum())
print(final_df["Season_Rainfall_mm"].describe())
print(final_df.nlargest(5, "Season_Rainfall_mm")[
    ["State", "Year", "Season", "Season_Rainfall_mm"]
])


final_df = final_df.dropna(subset=["Rice_Yield", "Season_Rainfall_mm",
                                    "Season_Temp_C"])
drop_states = {
    "Goa",
    "Andaman and Nicobar Islands",
    "Daman and Diu",
}

before = len(final_df)
final_df = final_df[
    ~final_df["State"].str.strip().isin(drop_states)
].copy()
after = len(final_df)

print(f"\n State removal")
print(f"Dropped {before - after} rows for {len(drop_states)} excluded states")
print(f"Remaining rows: {after}, states: {final_df['State'].nunique()}")
print("Dropped states:", sorted(drop_states))
print("Remaining states:", sorted(final_df["State"].unique()))


# 8. TARGET
# =========================================================
print("\n Building target ")
final_df = final_df.sort_values(["State", "Season", "Year"]).reset_index(drop=True)

final_df["Yield_Rolling_Avg"] = (
    final_df.groupby(["State", "Season"])["Rice_Yield"]
            .transform(lambda x: x.rolling(5, min_periods=2).mean())
)
final_df["Target_Risk"] = (
    final_df["Rice_Yield"] < final_df["Yield_Rolling_Avg"]
).astype("Int64")
final_df = final_df.drop(columns=["Yield_Rolling_Avg"])
final_df = final_df.dropna(subset=["Target_Risk"])
final_df["Target_Risk"] = final_df["Target_Risk"].astype(int)


# 9. FINAL REPORT 
# =========================================================
print("\n=== FINAL DATASET ===")
print("Shape        :", final_df.shape)
print("Columns      :", final_df.columns.tolist())
print("Seasons      :", final_df.groupby("Season").size().to_dict())
print("Year range   :", final_df["Year"].min(), "–", final_df["Year"].max())
print("States       :", final_df["State"].nunique())
print("Target dist  :", final_df["Target_Risk"].value_counts().to_dict())
print("\nHead:")
print(final_df.head())
print("\nDescribe:")
print(final_df.describe().round(2))

final_df.to_csv("FINAL_GSC_DATASET_ALLSEASONS.csv", index=False)
print("\n Saved: FINAL_GSC_DATASET_ALLSEASONS.csv")