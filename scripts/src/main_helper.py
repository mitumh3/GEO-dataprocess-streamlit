import asyncio
import os
import re

import streamlit as st
from data_process.process_func import get_data, merge_data
from dotenv import load_dotenv
from streamlit import session_state as cache

load_dotenv()
RESULT_PATH = os.getenv("result_path")
DATASET_PATH = os.getenv("dataset_path")


# Func to get existing datasets in dataset path:
def get_available_datasets_zip():
    try:
        available_datasets = set(
            name
            for file in os.listdir(DATASET_PATH)
            if file.endswith("_series_matrix.txt.gz")
            for name in re.findall(r"[^\W_]+", file)
            if name.startswith("GSE")
        )
    except BaseException:
        available_datasets = ""
    dataset_lst = list(available_datasets)
    cache.available_datasets = available_datasets
    return dataset_lst


def get_processed_dataset():
    try:
        available_datasets = set(name for name in os.listdir(RESULT_PATH))
    except BaseException:
        available_datasets = ""
    dataset_lst = list(available_datasets)
    cache.available_datasets = available_datasets
    return dataset_lst


# Func to initiate processing data
def run_processing(geo_id):
    data_exp, data_general_info, data_clin, data_extra, warn = asyncio.run(
        get_data(geo_id)
    )
    cache.data_exp = data_exp
    cache.data_general_info = data_general_info
    cache.data_clin = data_clin
    cache.warn = warn
    cache.data_extra = data_extra
    cache.processed = True


# Func to save files to csv
async def save_csv(data, path, message):
    try:
        data.to_csv(path, index=False)
        st.success(f"CSV file of {message} saved to {path}")
    except Exception as e:
        st.write(f"Error: {e}")


# Func to export data
async def export(data_exp, data_general_info, data_clin, geo_id):
    folder_path = f"{RESULT_PATH}/{geo_id}"
    file_path = f"{folder_path}/{geo_id}"
    with st.spinner("Exporting..."):
        if not os.path.exists(RESULT_PATH):
            print("\nFolder 'processed_data' created")
            os.mkdir(RESULT_PATH)
        if not os.path.exists(folder_path):
            print("\nFolder ", geo_id, " created")
            os.mkdir(folder_path)
        await asyncio.gather(
            save_csv(data_exp, f"{file_path}_expression.csv", "expression data"),
            save_csv(data_general_info, f"{file_path}_generalinfo.csv", "general info"),
            save_csv(data_clin, f"{file_path}_clinical.csv", "clinical data"),
        )
    st.balloons()


# Func to handle export click
def on_main_click():
    geo_id = cache.geo_id
    folder_path = f"{RESULT_PATH}/{geo_id}"
    if not os.path.exists(folder_path):
        data_exp = cache.data_exp
        data_general_info = cache.data_general_info
        data_clin = cache.data_clin

        cache.show_secondary = False
        asyncio.run(export(data_exp, data_general_info, data_clin, geo_id))
    else:
        cache.show_secondary = True


# Func to handle yes click of export
def on_yes_click():
    data_exp = cache.data_exp
    data_general_info = cache.data_general_info
    data_clin = cache.data_clin
    geo_id = cache.geo_id

    cache.show_secondary = False
    asyncio.run(export(data_exp, data_general_info, data_clin, geo_id))


# Func to handle no click of export
def on_no_click():
    cache.show_secondary = False


# Func to handle and display unused files
def handle_extra(data_extra, data_exp, data_clin):
    n_data_extra = len(data_extra)
    if n_data_extra != 0:
        st.subheader("Unused files")
        with st.form("data_extra_form"):
            columns = st.columns(n_data_extra)
            options = ["Expression data", "Clinical data", "None"]
            for i, col in enumerate(columns):
                file_name = list(data_extra)[i]
                with col:
                    st.write(file_name)
                    st.radio("Select options", options, key=file_name)
                    num_rows, num_cols = data_extra[file_name].shape
                    st.write(num_rows, " x ", num_cols)
                    st.write(data_extra[file_name].head())
            submit_button_extra = st.form_submit_button(label="Submit choice")
        if submit_button_extra:
            cache.button_extra = True
        if not cache.button_extra:
            st.write("Submit to add unused data to the chosen data")

        ## Action when submit button is clicked
        else:
            # st.write(data_extra)
            merge_data(data_extra, data_exp, data_clin)
            st.experimental_rerun()


def delete_cache(keep_lst=[]):
    # Delete all the items in Session state
    if cache.page != cache.current_page:
        keep_dict = {"current_page": cache.page}
        for key in cache.keys():
            if key in keep_lst:
                keep_dict[key] = cache[key]
            del cache[key]
        for key in keep_dict.keys():
            cache[key] = keep_dict[key]


def start_page(page_name, keep_cache_lst):
    cache.page = page_name
    if "current_page" not in cache:
        cache.current_page = ""
    delete_cache(keep_cache_lst)


# def delete_cache(keep_lst=[]):
#     # Delete all the items in Session state, except keep_lst
#     for key in cache.keys():
#         if key not in keep_lst:
#             del cache[key]
#         else:
#             keep_lst.remove(key)
#     print(keep_lst)
#     for remaining_item in keep_lst:
#         cache[remaining_item] = ""


# def start_page(page_name, keep_cache_lst):
#     cache.page = page_name
#     if "current_page" not in cache:
#         cache.current_page = ""

#     keep_cache_lst.append("current_page")
#     if cache.page != cache.current_page:
#         cache.current_page = page_name
#         delete_cache(keep_cache_lst)
