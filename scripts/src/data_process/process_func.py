import asyncio
import csv
import gzip
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as cache

from .process_helper import *

load_dotenv()
DATASET_PATH = os.getenv("dataset_path")


# Main function to get data with a ACCESSION ID input
async def get_data(accession_id):
    # Set Accession ID and folder path for saving files
    url_prefix, filelist = get_file_list(accession_id)

    # Start download and process files from GEO concurrently
    loop = asyncio.get_event_loop()
    coros = []
    for file_name, file_prefix in filelist.items():
        # Down file and process
        coro = process(url_prefix, file_name, file_prefix, loop)
        coros.append(coro)
    data_lst = await asyncio.gather(*coros)

    # Merge data
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
            except BaseException:
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


# Main function to process data
async def process(url1, file_name, file_prefix, loop):
    with st.spinner(f"Processing {file_name}"):
        url = f"{url1}{file_prefix}{file_name}"
        filepath = f"{DATASET_PATH}/{file_name}"
        # Download files
        await loop.run_in_executor(None, download_file, url, filepath)
        # Determine the largest number of columns
        with gzip.open(filepath, "rt") as f:
            reader = csv.reader(f, delimiter="\t")
            largest_column_count = max(len(row) for row in reader)

        # Create a list of column names
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

        # Display raw_data on tab1:
        tab1 = cache.tab1
        with tab1:
            st.subheader(file_name)
            try:
                st.write(df)
            except:
                st.warning("File too large, cannot display")

        check1 = df[df[0].str.contains("!series_matrix_table", na=False)]
        if check1.empty:
            data = {"filename": file_name, "extra": process_extra(df)}
        else:
            data = process_series_matrix(df)
    return data


# Main function to process input dataframe recognised as series matrix file
def process_series_matrix(df):
    data = {}
    data.update(extract_expression_data(df))
    data.update(extract_clinical_data(df))
    return data


# Main function to process input dataframe not recognised as series matrix file
def process_extra(df_extra):
    # Set first row as column names
    df_extra.columns = df_extra.iloc[0]
    data = df_extra.drop(0)
    return data


# Function to merge unused file with existing data - expression data or clinical data
def merge_data(data_extra, data_exp, data_clin):
    for i in range(len(data_extra)):
        file_name = list(data_extra)[0]
        if cache[file_name] == "Expression data":
            data_exp = pd.concat([data_exp, data_extra.pop(file_name)])
            data_exp.index.name = None
            data_exp = data_exp.dropna(how="all")
            data_exp = data_exp.reset_index(drop=True)
            # data_exp = data_exp.set_index(data_exp.columns[0])
            cache.data_exp = data_exp
            cache.extra = data_extra
        elif cache[file_name] == "Clinical data":
            clinical_extra = data_extra.pop(file_name)

            # Merge clinical data
            maxcheck2 = {}
            for old_column in data_clin.columns:
                for extra_column in clinical_extra.columns:
                    check2 = data_clin[
                        data_clin[old_column].isin(clinical_extra[extra_column])
                    ][old_column]
                    check2 = check2.dropna()
                    if len(check2) != 0:
                        if len(check2) > len(maxcheck2):
                            maxcheck2 = check2
                            print(old_column, extra_column)
                            col_1 = old_column
                            col_2 = extra_column
            st.write(old_column, extra_column)
            clinical = pd.merge(
                data_clin, clinical_extra, left_on=col_1, right_on=col_2, how="inner"
            )
            cache.data_clin = clinical
            cache.extra = data_extra
        else:
            data_extra.pop(file_name)
            cache.extra = data_extra
