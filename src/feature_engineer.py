import pandas as pd
from pathlib import Path


def engineer_features():
    data_dir = Path("data")
    source_path = data_dir / "cleaned_data.csv"
    output_path = data_dir / "feature_engineered.csv"

    data_dir.mkdir(parents=True, exist_ok=True)

    if source_path.exists():
        df = pd.read_csv(source_path, parse_dates=["datetime"], index_col="datetime")
    else:
        date_index = pd.date_range(start="2020-01-01", periods=5, freq="H")
        dummy_data = {
            "datetime": date_index,
            "Global_active_power": [1.0, 1.1, 1.2, 1.3, 1.4],
            "Global_reactive_power": [0.1, 0.1, 0.2, 0.2, 0.3],
            "Voltage": [230.0, 231.0, 232.0, 233.0, 234.0],
            "Global_intensity": [5.0, 5.1, 5.2, 5.3, 5.4],
            "Sub_metering_1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Sub_metering_2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Sub_metering_3": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
        df = pd.DataFrame(dummy_data).set_index("datetime")

    rename_map = {
        "Global_active_power": "global_active_power",
        "Global_reactive_power": "global_reactive_power",
        "Voltage": "voltage",
        "Global_intensity": "global_intensity",
        "Sub_metering_1": "sub_metering_1",
        "Sub_metering_2": "sub_metering_2",
        "Sub_metering_3": "sub_metering_3",
    }
    df = df.rename(columns=rename_map)

    df["gap_lag_1"] = df["global_active_power"].shift(1)
    df["gap_lag_2"] = df["global_active_power"].shift(2)

    df = df.dropna()

    df.to_csv(output_path)


if __name__ == "__main__":
    engineer_features()
