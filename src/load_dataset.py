from pathlib import Path
import pandas as pd


def load_csv_dataset(file_path):
    """
    Load a CSV dataset.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    data = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print(f"Number of records: {len(data)}")
    print(f"Columns: {list(data.columns)}")

    return data


if __name__ == "__main__":
    print("Dataset loader is ready.")
