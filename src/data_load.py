import glob
import os

import pandas as pd


def load_and_concat_excel_files(data_folder, file_pattern="*.csv"):
    """
    Reads all Excel files in the specified folder matching the file pattern
    and concatenates them vertically into a single DataFrame.

    Parameters:
        data_folder (str): Path to the folder containing the Excel files.
        file_pattern (str): Pattern to match Excel files (default: "*.xls").

    Returns:
        pd.DataFrame: Concatenated DataFrame from all Excel files.
    """
    # Create a full path pattern for glob
    path_pattern = os.path.join(data_folder, file_pattern)

    # Get a list of all matching files
    excel_files = glob.glob(path_pattern)
    print(f"Found {len(excel_files)} files.")

    # List to hold DataFrames
    df_list = []

    # Sort files numerically based on the prefix number
    excel_files.sort()

    for file in excel_files:
        try:
            print(f"Reading file: {file}")
            # Adjust engine if needed (e.g., engine="xlrd" for .xls files)
            df = pd.read_csv(file, low_memory=False)
            # Optionally, add a column for the source file if you want to track it
            df["source_file"] = os.path.basename(file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if df_list:
        concatenated_df = pd.concat(df_list, ignore_index=True)
        return concatenated_df
    else:
        print("No files were loaded successfully.")
        return pd.DataFrame()


# Example usage:
data_folder = "data/raw/fills/"  # Replace with your actual data folder
all_data = load_and_concat_excel_files(data_folder, file_pattern="*.csv")
print("Combined DataFrame shape:", all_data.shape)
print(all_data.head())
