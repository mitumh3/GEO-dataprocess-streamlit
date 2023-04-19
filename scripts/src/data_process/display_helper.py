import pandas as pd
import streamlit as st
from streamlit import session_state as cache


# Function to create submit button for ACCESSION ID input
def submit_geo_id(dataset_lst):
    with st.form("GEO_form"):
        id_input = st.text_input("Enter GEO Accession ID here:")

        dataset_lst.insert(0, None)
        with st.expander("Or choose existing datasets:"):
            id_radio = st.radio(
                label="Dataset choices",
                options=dataset_lst,
                horizontal=True,
                label_visibility="collapsed",
            )
        if id_radio != None:
            geo_id = id_radio
        else:
            geo_id = id_input

        cache.geo_id = geo_id
        submit_button_geo = st.form_submit_button(label="Submit")
    return geo_id, submit_button_geo


# Function to display data
def display_exp_and_extra(data_exp, data_general_info):
    num_rows1, num_cols1 = data_exp.shape
    num_rows2 = data_general_info.shape[0]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("General Information")
        st.write(num_rows2, " pieces of information")
        st.write(data_general_info)
    with col2:
        st.subheader("Expression Data Review")
        st.write(num_cols1 - 1, " samples with ", num_rows1, " genes")
        st.write(data_exp.head())


# Function to display column selection
def display_column_selection(options):
    st.subheader("Clinical Data")
    with st.form("data_clin_form"):
        feature_selection = st.multiselect("Choose columns to filter", options)
        submit_button_columnsel = st.form_submit_button(label="Submit choice")
    if submit_button_columnsel:
        cache.button_columnsel = "clicked"
    return feature_selection


def cooling_highlight(val):
    """
    >>> data_clin = data_clin.style.applymap(
                    cooling_highlight, subset=["!sample_title"]
                    )
    """
    color = "#ACE5EE" if val else "white"
    return f"background-color: {color}"


def display_warn(warn):
    if warn != None:
        st.warning(warn)


def display_clin(data_clin, warn, feature_selection):
    num_rows3, num_cols3 = data_clin.shape
    st.write(num_rows3, " samples with ", num_cols3, " features")
    if "clin_edit" not in cache:
        cache.clin_edit = False
    if cache.clin_edit == False:
        if "button_columnsel" not in cache or len(feature_selection) == 0:
            st.write(data_clin)
        else:
            st.write(data_clin[feature_selection])
    else:
        st.snow()
        with st.form("edit"):
            st.warning("You are in edit mode, BE CAREFUL!!!", icon="⚠️")
            if "button_columnsel" not in cache or len(feature_selection) == 0:
                edited_data = st.experimental_data_editor(data_clin, num_rows="dynamic")
            else:
                edited_data = st.experimental_data_editor(
                    data_clin[feature_selection], num_rows="dynamic"
                )
            save_button = st.form_submit_button(label="Save changes")
            cancel_button = st.form_submit_button(label="Cancel", type="primary")
        if save_button:
            cache.data_clin = edited_data
            cache.clin_edit = False
            st.experimental_rerun()
        elif cancel_button:
            cache.clin_edit = False
            st.experimental_rerun()
        display_warn(warn)
        st.stop()
    st.checkbox(label="Toggle edit mode", key="clin_edit")
    display_warn(warn)


# st.dataframe(
#     df1.style.applymap(cooling_highlight, subset=["Cooling inputs", "Cooling outputs"])
#     #  ,(heating_highlight, subset=['Heating inputs', 'Heating outputs'])
#     ,
#     height=530,
# )
