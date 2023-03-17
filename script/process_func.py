import gzip
import ssl
import urllib.request
import asyncio
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from GEOparse import utils as GEO
import os

load_dotenv()
dataset_path = os.getenv("dataset_path")
result_path = os.getenv("result_path")

async def get_data(accession_ID):
    # Set Accession ID and folder path for saving files
    context = ssl._create_unverified_context()
    url1 = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{accession_ID[:-3]}nnn/{accession_ID}/"
    geo_folders = ["matrix/", "miniml/", "soft/", "suppl/"]
    filelist = {}
    GEO.mkdir_p(dataset_path)

    for file_prefix in geo_folders:
        url2 = f"{url1}/{file_prefix}"

        # Find all links in the HTML content
        links = BeautifulSoup(urllib.request.urlopen(url2, context=context).read(), "html.parser").find_all("a")

        # Print the file names
        for link in links:
            file_name = link.get("href")
            if (file_name.endswith(".txt.gz") 
                    or file_name.endswith(".tsv.gz") 
                    or file_name.endswith(".csv.gz")):
                filelist[file_name] = file_prefix

    loop = asyncio.get_event_loop()
    coros = []
    for file_name, file_prefix in filelist.items():
        # Down file and process
        coro = process(url1, file_name, file_prefix, loop) 
        coros.append(coro)
    data_lst = await asyncio.gather(*coros)
    # loop.close()

    expression_data = pd.DataFrame()
    gereral_data = {}
    clinical = pd.DataFrame()
    data_extra = {}
    for data in data_lst:
        # Handle data
        if "type" in data:
            try:
                exp = data["expression"].T
                exp = exp.set_axis(exp.iloc[0], axis=1).iloc[1:]
            except:
                exp = pd.DataFrame()
            gen = data["general"]
            # Add elements in data to existing expression, general and clinical
            for feature, info in gen.items():
                if feature not in gereral_data:
                    gereral_data[feature] = info

            expression_data = pd.concat([expression_data, exp])
            gereral_data = pd.DataFrame(gereral_data)
            clinical = pd.concat([clinical, data["clinical"]]).reset_index(drop=True)
        else:
            # Unused files
            data_extra[data["filename"]] = data["extra"]

    expression_data = expression_data.T.reset_index()
    for i in [expression_data, gereral_data, clinical]:
        i.dropna(axis=1, how="all", inplace=True)
        i.dropna(axis=0, how="all", inplace=True)

    # # Check dataframe integrity
    warn = check_dataframe_integrity(clinical)
    return expression_data, gereral_data, clinical, data_extra, warn

def download_file(url, filepath):
    GEO.download_from_url(url, destination_path=filepath)


async def process(url1, file_name, file_prefix, loop):
    url = f"{url1}{file_prefix}{file_name}"
    filepath = f"{dataset_path}/{file_name}"
    # Download files 
    await loop.run_in_executor(None, download_file, url, filepath)
    # Read files
    with gzip.open(filepath, "rt") as f:
        largest_column_count = max(len(line.split("\t")) for line in f)
    column_names = [i for i in range(0, largest_column_count)]

    # Read file as dataframe
    df = pd.read_csv(
        filepath,
        compression="gzip",
        header=None,
        keep_default_na=True,
        skip_blank_lines=False,
        delimiter="\t",
        quotechar='"',
        on_bad_lines="skip",
        names=column_names,
        low_memory=False,
    )

    check1 = df[df[0].str.contains("!series_matrix_table", na=False)]
    if check1.empty:
        data = {"filename":file_name, "extra":process_extra(df)}
    else:
        data = process_series_matrix(df)
    return data


def process_series_matrix(df):
    drop_list = ["!sample_relation", "!sample_supplementary_file"]
    data = {"type": "series_matrix"}
    # Cut out expression data based on !series_matrix_table_begin and !series_matrix_table_end
    try:
        start, end = df[df[0].str.contains("!series_matrix_table", na=False)].index
        expression_data = df.iloc[start + 1 : end]
        # Rename columns as sample id and reindex rows
        expression_data = expression_data.rename(columns=expression_data.iloc[0, :]).iloc[1:, :].reset_index(drop=True)
        data["expression"] = expression_data
    except:
        data["expression"] = pd.DataFrame()
        print("No expression data")

    # Cut out clinical data based on string "!Sample"
    df1 = df[df[0].str.contains("!Sample", na=False)]
    df1 = df1.set_index(0)
    df1.index.name = None
    df1.columns = range(df1.shape[1])
    df1 = df1.dropna(how="all")
    # Transpose and Rename columns as sample features and reindex rows
    clinical_data = df1.transpose().set_axis(df1.index, axis=1)
    clinical_data = clinical_data.rename(columns={col: col.lower() for col in clinical_data.columns})

    # Drop columns based on drop_list
    dropped_cols = [col for col in drop_list if col in clinical_data.columns]
    clinical_data.drop(dropped_cols, axis=1, inplace=True)
    unused_cols = [col for col in drop_list if col not in dropped_cols]

    print(f"\nUnused columns from drop_list: {unused_cols}")

    # Fix duplicate column names
    class renamer:
        def __init__(self):
            self.d = dict()

        def __call__(self, x):
            if x not in self.d:
                self.d[x] = 0
                return x
            else:
                self.d[x] += 1
                return "%s_%d" % (x, self.d[x])

    clinical_data.rename(columns=renamer(), inplace=True)

    # Clear column with one value from dataframe
    sample_general_info = {"Feature": [], "Info": []}
    column_take = []
    for i in range(len(clinical_data.columns)):
        if len(set(clinical_data.iloc[:, i])) == 1:
            sample_general_info["Feature"].append(clinical_data.columns[i])
            sample_general_info["Info"].append(list(set(clinical_data.iloc[:, i])).pop())
        else:
            column_take.append(i)
    clean_clinical_data = clinical_data.iloc[:, column_take]

    # Create sample general info table
    data["general"] = sample_general_info

    # Take out data from one cells with split ": "
    cleaned_clinical_data = pd.DataFrame()
    for t in range(clean_clinical_data.shape[0]):
        df2 = pd.DataFrame(clean_clinical_data.loc[t]).T
        # print(df2)
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
        # clean_clinical_data2=clean_clinical_data.iloc[:,column_take2]
        new_feature = new_feature.rename(columns={col: col.lower() for col in new_feature.columns})
        df2 = df2[column_take2]
        try:
            index_no = df2.columns.get_loc("!sample_geo_accession")
            df2 = pd.concat([df2.iloc[:, : index_no + 1], new_feature, df2.iloc[:, index_no + 1 :]], axis=1)
        except:
            print("Cannot find !sample_geo_accession")

        cleaned_clinical_data = pd.concat([cleaned_clinical_data, df2])
    for i in cleaned_clinical_data.columns:
        if cleaned_clinical_data[i].isna().all():
            cleaned_clinical_data = cleaned_clinical_data.drop(i, axis=1)
    data["clinical"] = cleaned_clinical_data
    return data


def process_extra(df):
    # Set first row as column names
    df.columns = df.iloc[0]
    data = df.drop(0)
    return data


def check_dataframe_integrity(df):
    """Check for various integrity issues in a pandas DataFrame."""
    issues_found = False

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        try:
            warn = f"WARNING: Missing values found in the following columns: {', '.join(missing_values[missing_values>0].index)}"
            print(warn)
        except:
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


def merge_data(data_extra, exp_data, clin_data):
    for i in range(len(data_extra)):
        file_name = list(data_extra)[0]
        if st.session_state[file_name] == "Expression data":
            exp_data = pd.concat([exp_data, data_extra.pop(file_name)])
            exp_data.index.name = None
            exp_data = exp_data.dropna(how="all")
            exp_data = exp_data.reset_index(drop=True)
            # exp_data = exp_data.set_index(exp_data.columns[0])
            st.session_state.exp_data = exp_data
            st.session_state.extra = data_extra
        elif st.session_state[file_name] == "Clinical data":
            clinical_extra = data_extra.pop(file_name)

            # Merge clinical data
            maxcheck2 = {}
            for i in clin_data.columns:
                for t in clinical_extra.columns:
                    check2 = clin_data[clin_data[i].isin(clinical_extra[t])][i]
                    check2 = check2.dropna()
                    if len(check2) != 0:
                        if len(check2) > len(maxcheck2):
                            maxcheck2 = check2
                            print(i, t)
                            col_1 = i
                            col_2 = t
            st.write(i, t)
            clinical = pd.merge(clin_data, clinical_extra, left_on=col_1, right_on=col_2, how="inner")
            st.session_state.clin_data = clinical
            st.session_state.extra = data_extra
        else:
            data_extra.pop(file_name)
            st.session_state.extra = data_extra
