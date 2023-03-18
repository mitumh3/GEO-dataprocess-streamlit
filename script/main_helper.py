import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

from process_func import merge_data

load_dotenv()
RESULT_PATH = os.getenv("result_path")


async def save_csv(data, path, message):
    try:
        data.to_csv(path, index=False)
        st.write(f"# ***CSV file of {message} saved to {path}")
    except Exception as e:
        st.write(f"Error: {e}")


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


def on_main_click(exp_data, general_info, clin_data, geo_id):
    folder_path = f"{RESULT_PATH}/{geo_id}"
    if not os.path.exists(folder_path):
        st.session_state.show_secondary = False
        asyncio.run(export(exp_data, general_info, clin_data, geo_id))
    else:
        st.session_state.show_secondary = True


def on_yes_click(exp_data, general_info, clin_data, geo_id):
    st.session_state.show_secondary = False
    asyncio.run(export(exp_data, general_info, clin_data, geo_id))


def on_no_click():
    st.session_state.show_secondary = False


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
