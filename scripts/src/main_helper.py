import asyncio
import os
import re

import streamlit as st
from dotenv import load_dotenv

from process_func import get_data, merge_data

load_dotenv()
RESULT_PATH = os.getenv("result_path")
DATASET_PATH = os.getenv("dataset_path")


# Func to get existing datasets
def get_available_datasets():
    try:
        available_datasets = set(
            word
            for file in os.listdir(DATASET_PATH)
            if file.endswith("_series_matrix.txt.gz")
            for word in re.findall(r"[^\W_]+", file)
            if word.startswith("GSE")
        )
    except BaseException:
        available_datasets = ""
    st.session_state.available_datasets = available_datasets
    dataset_list = (
        " | ".join(str(item) for item in st.session_state.available_datasets)
        if st.session_state.available_datasets
        else None
    )
    return dataset_list


# Func to initiate processing data
def run_process(geo_id):
    exp_data, general_info, clin_data, data_extra, warn = asyncio.run(get_data(geo_id))
    st.session_state.exp_data = exp_data
    st.session_state.general_info = general_info
    st.session_state.clin_data = clin_data
    st.session_state.warn = warn
    st.session_state.extra = data_extra
    st.session_state.processed = True


# Func to save files to csv
async def save_csv(data, path, message):
    try:
        data.to_csv(path, index=False)
        st.write(f"# ***CSV file of {message} saved to {path}")
    except Exception as e:
        st.write(f"Error: {e}")


# Func to export data
async def export(exp_data, general_info, clin_data, geo_id):
    folder_path = f"{RESULT_PATH}/{geo_id}"
    file_path = f"{folder_path}/{geo_id}"
    with st.spinner("Exporting..."):
        if not os.path.exists("../processed_data/"):
            print("\nFolder 'processed_data' created")
            os.mkdir("../processed_data/")
        if not os.path.exists(folder_path):
            print("\nFolder ", geo_id, " created")
            os.mkdir(folder_path)
        await asyncio.gather(
            save_csv(exp_data, f"{file_path}_expression.csv", "expression data"),
            save_csv(general_info, f"{file_path}_generalinfo.csv", "general info"),
            save_csv(clin_data, f"{file_path}_clinical.csv", "clinical data"),
        )
        st.success("Files exported!")


# Func to handle export click
def on_main_click(exp_data, general_info, clin_data, geo_id):
    folder_path = f"{RESULT_PATH}/{geo_id}"
    if not os.path.exists(folder_path):
        st.session_state.show_secondary = False
        asyncio.run(export(exp_data, general_info, clin_data, geo_id))
    else:
        st.session_state.show_secondary = True


# Func to handle yes click of export
def on_yes_click(exp_data, general_info, clin_data, geo_id):
    st.session_state.show_secondary = False
    asyncio.run(export(exp_data, general_info, clin_data, geo_id))


# Func to handle no click of export
def on_no_click():
    st.session_state.show_secondary = False


# Func to handle and display unused files
def handle_extra(data_extra, exp_data, clin_data):
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
            st.session_state.button_extra = True
        if not st.session_state.button_extra:
            st.write("Submit to add unused data to the chosen data")

        #     # Action when submit button is clicked
        else:
            # st.write(data_extra)
            merge_data(data_extra, exp_data, clin_data)
            st.experimental_rerun()
