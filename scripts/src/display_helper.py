import streamlit as st
from streamlit import session_state as cache


# Function to create submit button for ACCESSION ID input
def submit_geo_id(dataset_lst):
    with st.form("GEO_form"):
        geo_id = st.text_input("Enter GEO Accession ID here:")
        cache.geo_id = geo_id
        submit_button_geo = st.form_submit_button(label="Submit")

        dataset_str = " | ".join(dataset_lst)
        st.write(">>>>> Dataset(s) found in your folder:")
        st.write(dataset_str)
    return geo_id, submit_button_geo


# Function to display data
def display_data(exp_data, general_info, clin_data):
    num_rows1, num_cols1 = exp_data.shape
    num_rows2 = general_info.shape[0]
    num_rows3, num_cols3 = clin_data.shape
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("General Information")
        st.write(num_rows2, " pieces of information")
        st.write(general_info)
    with col2:
        st.subheader("Expression Data Review")
        st.write(num_cols1 - 1, " samples with ", num_rows1, " genes")
        st.write(exp_data.head())
    st.subheader("Clinical Data")
    st.write(num_rows3, " samples with ", num_cols3, " features")
    return st.form("clin_data_form"), clin_data


# Function to display column selection
def display_column_selection(form, clin_data, warn):
    with form:
        feature_selection = st.multiselect(
            "Choose columns to filter and submit", clin_data.columns
        )
        submit_button_columnsel = st.form_submit_button(label="Submit choice")
    if submit_button_columnsel:
        cache.button_columnsel = "clicked"
    if "button_columnsel" not in cache or len(feature_selection) == 0:
        st.write(clin_data)
        st.write(warn)
    else:
        st.write(clin_data[feature_selection])
