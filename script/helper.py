import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

geoID = ""
dataset_path = os.getenv("dataset_path")
result_path = os.getenv("result_path")
folder_path = f"{result_path}/{geoID}"
file_path = f"{folder_path}/{geoID}"


def export(exp_data, general_info, clin_data):
    with st.spinner("Exporting..."):
        if not os.path.exists("../processed_data/"):
            print("\nFolder 'processed_data' created")
            os.mkdir("../processed_data/")
        if not os.path.exists(folder_path):
            print("\nFolder ", geoID, " created")
            os.mkdir(folder_path)
        try:
            exp_data.to_csv(file_path + "_expression.csv", index=False)
            st.write(f"# ***CSV file of expression data saved to {folder_path}")
        except Exception as e:
            st.write(f"Error: {e}")
        try:
            general_info.to_csv(file_path + "_generalinfo.csv", index=False)
            st.write(f"# ***CSV file of general info saved to {folder_path}")
        except Exception as e:
            st.write(f"Error: {e}")
        try:
            clin_data.to_csv(file_path + "_clinical.csv", index=False)
            st.write(f"# ***CSV file of clinical data saved to {folder_path}")
        except Exception as e:
            st.write(f"Error: {e}")
        st.success("Files exported!")


def on_main_click(exp_data, general_info, clin_data):
    if not os.path.exists(file_path + "_clinical.csv"):
        st.session_state.show_secondary = False
        export(exp_data, general_info, clin_data)
    else:
        st.session_state.show_secondary = True


def on_yes_click(exp_data, general_info, clin_data):
    st.session_state.show_secondary = False
    export(exp_data, general_info, clin_data)


def on_no_click():
    st.session_state.show_secondary = False
