import pandas as pd
from pathlib import Path


def clean_data():
    RAW_PATH = Path("data/household_power_consumption.txt")
    OUT_PATH = Path("data/cleaned_data.csv")

    # Read dataset
    df = pd.read_csv(
        RAW_PATH,
        sep=';',
        na_values='?',
        low_memory=False
    )

    # Combine date and time into a datetime column
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df = df.drop(columns=["Date", "Time"])

    # Convert numeric columns
    num_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Set datetime index and resample to hourly mean
    df = df.set_index("datetime").sort_index()
    df = df.resample("h").mean().dropna(how="all")

    # Save cleaned dataset
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH)
    print(f"✅ Cleaned dataset saved to {OUT_PATH}")


if __name__ == "__main__":
    clean_data()
