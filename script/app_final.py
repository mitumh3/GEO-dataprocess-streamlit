import asyncio
import os
import re

import streamlit as st
from dotenv import load_dotenv

from helper import handle_extra, on_main_click, on_no_click, on_yes_click
from process_func import get_data

load_dotenv()
DATASET_PATH = os.getenv("dataset_path")


async def main():
    # Add title
    st.set_page_config(page_title="Process Data")
    st.session_state.initial = st.session_state
    # List of available datasets in folder
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
    avail_datasets = (
        " | ".join(str(item) for item in st.session_state.available_datasets)
        if st.session_state.available_datasets
        else None
    )

    # Add a text input form
    with st.form("GEO_form"):
        geo_id = st.text_input("Enter GEO Accession ID here:")
        # Every form must have a submit button.
        submit_button_geo = st.form_submit_button(label="Submit")
        st.write(">>>>> Dataset(s) found in your folder:")
        st.write(avail_datasets)
    # Only if the submit button is clicked once that the code continue
    if submit_button_geo:
        st.session_state.button_GEO = "clicked"
        st.session_state.processed = False
        st.session_state.button_extra = False
    if "button_GEO" not in st.session_state:
        st.write("Click Submit to continue")
        st.stop()

    # Results after submit
    else:
        st.write("You entered:", geo_id)
        st.write(
            "\n<h1 style='text-align: center;'>" + geo_id + "</h1>\n",
            unsafe_allow_html=True,
        )
        # st.session_state
        # Bind 3 main dataframes to variables
        if not st.session_state.processed:
            with st.spinner("Processing..."):
                exp_data, general_info, clin_data, data_extra, warn = await get_data(geo_id)
                st.session_state.exp_data = exp_data
                st.session_state.general_info = general_info
                st.session_state.clin_data = clin_data
                st.session_state.warn = warn
                st.session_state.extra = data_extra
                st.session_state.processed = True

        exp_data = st.session_state.exp_data
        general_info = st.session_state.general_info
        clin_data = st.session_state.clin_data
        warn = st.session_state.warn
        data_extra = st.session_state.extra

        # Display unused files with select radio
        handle_extra(data_extra, exp_data, clin_data)

        # Display data
        num_rows1, num_cols1 = exp_data.shape
        num_rows2 = general_info.shape[0]
        num_rows3, num_cols3 = clin_data.shape
        col1, col2 = st.columns(2)
        # First Dataframe
        with col1:
            st.subheader("General Information")
            st.write(num_rows2, " pieces of information")
            st.write(general_info)
            # First Dataframe
        with col2:
            st.subheader("Expression Data Review")
            st.write(num_cols1 - 1, " samples with ", num_rows1, " genes")
            st.write(exp_data.head())
            # Third Dataframe
        st.subheader("Clinical Data")
        st.write(num_rows3, " samples with ", num_cols3, " features")
        # Create a column selection field and a submit button
        with st.form("clin_data_form"):
            feature_selection = st.multiselect(
                "Choose columns to filter and submit", clin_data.columns
            )
            submit_button_columnsel = st.form_submit_button(label="Submit choice")
            # Every form must have a submit button.
        if submit_button_columnsel:
            st.session_state.button_columnsel = "clicked"
        if "button_columnsel" not in st.session_state or len(feature_selection) == 0:
            st.write(clin_data)
            st.write(warn)
        else:
            st.write(clin_data[feature_selection])

    # # Submit button for exporting files
    if "show_secondary" not in st.session_state:
        st.session_state.show_secondary = False
    if not st.session_state.show_secondary:
        st.button(
            "Export files",
            on_click=on_main_click,
            args=(exp_data, general_info, clin_data, geo_id),
        )
    if st.session_state.show_secondary:
        # if os.path.exists(file_path + "_clinical"):
        st.warning(
            "Result files are found in the folder, overwrite existing files?\n"
            "Remember to close files before overwriting"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.button(
                "Yes",
                on_click=on_yes_click,
                args=(exp_data, general_info, clin_data, geo_id),
            )
        with col2:
            st.button("No", on_click=on_no_click)


if __name__ == "__main__":
    asyncio.run(main())
