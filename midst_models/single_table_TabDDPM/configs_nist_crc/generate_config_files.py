import pandas as pd
import os
import numpy as np
from typing import Dict, List, Set, Union
import json

data_path = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/NIST_Red-Team_Problems1-24_v2/"

# number of unique values at which we consider the feature continuous vs. discrete.
CONTINUOUS_THRESHOLD = 45

probs = {
    "QI1": [
        "11_AIM_e1_25f_QID1",
        "13_AIM_e10_25f_QID1",
        "17_TVAE_25f_QID1",
        "19_CELL_SUPPRESSION_25f_QID1",
        "1_SYNTHPOP_25f_QID1",
        "15_ARF_25f_QID1",
        "23_RANKSWAP_25f_QID1",
        "5_MST_e1_25f_QID1",
        "7_MST_e10_25f_QID1",
    ],
    "QI2": [
        "12_AIM_e1_25f_QID2",
        "14_AIM_e10_25f_QID2",
        "16_ARF_25f_QID2",
        "18_TVAE_25f_QID2",
        "20_CELL_SUPPRESSION_25f_QID2",
        "24_RANKSWAP_25f_QID2",
        "2_SYNTHPOP_25f_QID2",
        "6_MST_e1_25f_QID2",
        "8_MST_e10_25f_QID2",
    ]
}

probs50f = {
    "QI1": [
        "21_CELL_SUPPRESSION_50f_QID1",
        "3_SYNTHPOP_50f_QID1",
        "9_MST_e10_50f_QID1",
    ],
    "QI2": [
        "22_CELL_SUPPRESSION_50f_QID2",
        "4_SYNTHPOP_50f_QID2",
        "10_MST_e10_50f_QID2",
    ]
}



all_data = pd.DataFrame()



def load_dataset(data_path: str, folder: str, filename: str) -> pd.DataFrame:
    file_path = os.path.join(data_path, f"{filename}_Deid.csv")
    try:
        df = pd.read_csv(file_path)
        # Add metadata columns to identify dataset origin
        df['file_origin'] = filename
        df['qi_folder'] = folder
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def load_all_datasets(data_path: str, probs: Dict[str, List[str]], probs50f: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    all_dfs = {}

    # Load all 25f datasets
    for folder, filenames in probs.items():
        for filename in filenames:
            key = f"{folder}_{filename}"
            all_dfs[key] = load_dataset(data_path, folder, filename)
            print(f"Loaded {key}: {all_dfs[key].shape}")

    # Load all 50f datasets
    for folder, filenames in probs50f.items():
        for filename in filenames:
            key = f"{folder}_{filename}"
            all_dfs[key] = load_dataset(data_path, folder, filename)
            print(f"Loaded {key}: {all_dfs[key].shape}")

    return all_dfs


def combine_datasets(all_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Get all non-empty DataFrames
    valid_dfs = [df for df in all_dfs.values() if not df.empty]

    if not valid_dfs:
        print("No valid DataFrames to combine!")
        return pd.DataFrame()

    # Combine all DataFrames
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    return combined_df


def get_column_domains(df: pd.DataFrame) -> Dict[str, Union[List, Set, Dict]]:
    domains = {}

    # Skip metadata columns we added
    skip_cols = ['file_origin', 'qi_folder', 'target']

    for col in df.columns:
        if col in skip_cols:
            continue

        col_type = df[col].dtype
        unique_vals = df[col].nunique()

        # For numeric columns, get min and max
        if pd.api.types.is_numeric_dtype(col_type):
            min_val = df[col].min()
            max_val = df[col].max()
            domains[col] = {
                'type': str(col_type),
                'min': min_val,
                'max': max_val,
                'unique_values': unique_vals,
                'has_nulls': df[col].isna().any(),
                'null_count': df[col].isna().sum()
            }

        # For categorical columns, get unique values if not too many
        elif pd.api.types.is_categorical_dtype(col_type) or pd.api.types.is_object_dtype(col_type):
            print("SHOULD NOT BE ANY NONNUMERICAL")
            domains[col] = {
                'type': str(col_type),
                'unique_values': unique_vals,
                'has_nulls': df[col].isna().any(),
                'null_count': df[col].isna().sum()
            }

        # For other column types
        else:
            domains[col] = {
                'type': str(col_type),
                'unique_values': df[col].nunique(),
                'has_nulls': df[col].isna().any(),
                'null_count': df[col].isna().sum()
            }

        # Only include actual values if there aren't too many
        if unique_vals <= 20:  # Arbitrary limit to prevent huge outputs
            domains[col]['values'] = sorted(df[col].dropna().unique().tolist())

    return domains


def analyze_dataset_structure(all_dfs: Dict[str, pd.DataFrame]) -> Dict:
    analysis = {
        'dataset_shapes': {},
        'columns_by_dataset': {},
        'column_occurrence': {},
        'f50_only_columns': set(),
    }

    # Track columns in each dataset
    all_columns = set()
    for key, df in all_dfs.items():
        if df.empty:
            continue

        # Remove metadata columns for analysis
        cols = [c for c in df.columns if c not in ['file_origin', 'qi_folder', 'target']]
        analysis['dataset_shapes'][key] = df.shape
        analysis['columns_by_dataset'][key] = cols
        all_columns.update(cols)

        # Track column occurrence
        for col in cols:
            if col not in analysis['column_occurrence']:
                analysis['column_occurrence'][col] = []
            analysis['column_occurrence'][col].append(key)

    f25_columns = set()
    f50_columns = set()
    for key, cols in analysis['columns_by_dataset'].items():
        if '25f' in key:
            f25_columns.update(cols)
        if '50f' in key:
            f50_columns.update(cols)
    analysis['f50_only_columns'] = f50_columns - f25_columns

    return analysis

def make_domain_file(data, domains):
    domain = {}
    for col in domains.keys():
        domain[col] = {
            'size': domains[col]['unique_values'],
            'type': 'discrete' if domains[col]['type'] == 'int64' or domains[col]['unique_values'] < CONTINUOUS_THRESHOLD else 'continuous',
        }

    with open('crc_data_domain.json', 'w') as f:
        json.dump(domain, f)




# Main execution
def main():
    all_dfs = load_all_datasets(data_path, probs, probs50f)
    structure_analysis = analyze_dataset_structure(all_dfs)
    all_data = combine_datasets(all_dfs)
    domains = get_column_domains(all_data)

    sample_cols = list(domains.keys())[:]
    for col in sample_cols:
        print(f"\n{col}:")
        for k, v in domains[col].items():
            print(f"  {k}: {v}")

    make_domain_file(all_data, domains)



if __name__ == "__main__":
    main()