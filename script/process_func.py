import asyncio
import gzip
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from process_helper import (check_dataframe_integrity, download_file, extract_clinical_data,
                            extract_expression_data, get_file_list)

load_dotenv()
DATASET_PATH = os.getenv("dataset_path")


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


async def process(url1, file_name, file_prefix, loop):
    url = f"{url1}{file_prefix}{file_name}"
    filepath = f"{DATASET_PATH}/{file_name}"
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
        data = {"filename": file_name, "extra": process_extra(df)}
    else:
        data = process_series_matrix(df)
    return data


# Main function that calls the other functions to process input dataframe
def process_series_matrix(df):
    data = {}
    data.update(extract_expression_data(df))
    data.update(extract_clinical_data(df))
    return data


def process_extra(df_extra):
    # Set first row as column names
    df_extra.columns = df_extra.iloc[0]
    data = df_extra.drop(0)
    return data


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
            for old_column in clin_data.columns:
                for extra_column in clinical_extra.columns:
                    check2 = clin_data[clin_data[old_column].isin(clinical_extra[extra_column])][
                        old_column
                    ]
                    check2 = check2.dropna()
                    if len(check2) != 0:
                        if len(check2) > len(maxcheck2):
                            maxcheck2 = check2
                            print(old_column, extra_column)
                            col_1 = old_column
                            col_2 = extra_column
            st.write(old_column, extra_column)
            clinical = pd.merge(
                clin_data, clinical_extra, left_on=col_1, right_on=col_2, how="inner"
            )
            st.session_state.clin_data = clinical
            st.session_state.extra = data_extra
        else:
            data_extra.pop(file_name)
            st.session_state.extra = data_extra
