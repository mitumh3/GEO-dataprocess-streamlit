import os
import ssl
import urllib.request

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from GEOparse import utils as GEO

load_dotenv()
DATASET_PATH = os.getenv("dataset_path")


# Function to get file list of GEO from FTP server
def get_file_list(accession_id):
    context = ssl._create_unverified_context()
    url_prefix = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{accession_id[:-3]}nnn/{accession_id}/"
    GEO_FOLDERS = ["matrix/", "miniml/", "soft/", "suppl/"]
    filelist = {}
    for file_prefix in GEO_FOLDERS:
        url_folder = f"{url_prefix}/{file_prefix}"
        # Find all links in the HTML content, and add file names to filelist
        links = BeautifulSoup(
            urllib.request.urlopen(url_folder, context=context).read(), "html.parser"
        ).find_all("a")
        for link in links:
            file_name = link.get("href")
            if (
                file_name.endswith(".txt.gz")
                or file_name.endswith(".tsv.gz")
                or file_name.endswith(".csv.gz")
            ):
                filelist[file_name] = file_prefix
    return url_prefix, filelist


# Function to download files from GEO databse
def download_file(url, filepath):
    GEO.mkdir_p(DATASET_PATH)
    GEO.download_from_url(url, destination_path=filepath)


# Function to extract expression data from input dataframe
def extract_expression_data(df):
    data = {"type": "series_matrix"}
    try:
        start, end = df[df[0].str.contains("!series_matrix_table", na=False)].index
        expression_data = df.iloc[start + 1 : end]
        expression_data = (
            expression_data.rename(columns=expression_data.iloc[0, :])
            .iloc[1:, :]
            .reset_index(drop=True)
        )
        data["expression"] = expression_data
    except BaseException:
        data["expression"] = pd.DataFrame()
        print("No expression data")
    return data


# Function to extract clinical data from input dataframe
def extract_clinical_data(df):
    drop_list = ["!sample_relation", "!sample_supplementary_file"]
    df1 = df[df[0].str.contains("!Sample", na=False)]
    df1 = df1.set_index(0)
    df1.index.name = None
    df1.columns = range(df1.shape[1])
    df1 = df1.dropna(how="all")
    clinical_data = df1.transpose().set_axis(df1.index, axis=1)
    clinical_data = clinical_data.rename(
        columns={col: col.lower() for col in clinical_data.columns}
    )

    # Drop columns based on drop_list
    dropped_cols = [col for col in drop_list if col in clinical_data.columns]
    clinical_data.drop(dropped_cols, axis=1, inplace=True)

    # Fix duplicate column names
    class Renamer:
        def __init__(self):
            self.d = {}

        def __call__(self, x):
            if x not in self.d:
                self.d[x] = 0
                return x
            self.d[x] += 1
            return f"{x}_{self.d[x]}"

    clinical_data.rename(columns=Renamer(), inplace=True)

    # Remove single value columns and create sample general info table
    sample_general_info = {"Feature": [], "Info": []}
    column_take = []
    for col_idx, col_name in enumerate(clinical_data.columns):
        if len(set(clinical_data.iloc[:, col_idx])) == 1:
            sample_general_info["Feature"].append(clinical_data.columns[col_idx])
            sample_general_info["Info"].append(list(set(clinical_data.iloc[:, col_idx])).pop())
        else:
            column_take.append(col_idx)
    cleaned_clinical_data = clinical_data.iloc[:, column_take]

    # Split values in cells with ": " and create new columns
    cleaned_clinical_data = split_values(cleaned_clinical_data)

    return {"general": sample_general_info, "clinical": cleaned_clinical_data}


# Function to split values in cells with ":"
def split_values(data):
    new_data = pd.DataFrame()
    for t in range(data.shape[0]):
        df2 = pd.DataFrame(data.loc[t]).transpose()
        column_take2 = []
        new_feature = pd.DataFrame()
        for i in df2.columns:
            new = df2[i].str.split(": ", n=1, expand=True)
            if len(new.columns) == 2:
                if len(set(new[0])) == 1:
                    new_feature[np.unique(new[0])[0]] = new[1]
                else:
                    print("Something wrong with separating string process at column: ", i)
            else:
                column_take2.append(i)
        new_feature = new_feature.rename(columns={col: col.lower() for col in new_feature.columns})
        df2 = df2[column_take2]
        try:
            index_no = df2.columns.get_loc("!sample_geo_accession")
            df2 = pd.concat(
                [df2.iloc[:, : index_no + 1], new_feature, df2.iloc[:, index_no + 1 :]],
                axis=1,
            )
        except BaseException:
            print("Cannot find !sample_geo_accession")

        new_data = pd.concat([new_data, df2])
    for i in new_data.columns:
        if new_data[i].isna().all():
            new_data = new_data.drop(i, axis=1)

    return new_data


# Function to check dataframe integrity
def check_dataframe_integrity(df):
    """Check for various integrity issues in a pandas DataFrame."""
    issues_found = False

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        try:
            warn = f"WARNING: Missing values found in the following columns: {', '.join(missing_values[missing_values>0].index)}"
            print(warn)
        except BaseException:
            print("Columns only have index not name")
            print(missing_values[missing_values > 0].index)
        for col in missing_values[missing_values > 0].index:
            print(f"  Missing values found in column {col} at positions:")
            positions = np.where(df[col].isnull())[0]
            print(f"  {positions}")
        issues_found = True

    # Check for duplicates
    num_duplicates = df.duplicated().sum()
    if num_duplicates:
        print(f"Warning: {num_duplicates} duplicate(s) found in the DataFrame")
        issues_found = True

    # Check for negative or zero values in numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if (df[col] <= 0).any():
            print(f"Warning: Negative or zero values found in column {col}")
            issues_found = True

    # Check for outliers in numeric columns
    for col in numeric_cols:
        if df[col].dtype != "object":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                print(f"Warning: {len(outliers)} outlier(s) found in column {col}")
                issues_found = True

    if not issues_found:
        print("No integrity issues found in DataFrame")
    else:
        return warn
