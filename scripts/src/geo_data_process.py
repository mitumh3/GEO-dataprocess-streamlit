import streamlit as st
from display_helper import *
from main_helper import *

# from plotting import *


def geo_data_process():
    st.write("# Process Data 👋")
    st.session_state.button_extra = False

    # display datasets found
    dataset_list = get_available_datasets()
    geo_id, submit_button_geo = submit_geo_id(dataset_list)

    # continue only if submit button was clicked
    if submit_button_geo:
        st.session_state.button_GEO = "clicked"
        st.session_state.processed = False
        st.session_state.button_extra = False
    if "button_GEO" not in st.session_state:
        st.write("Click Submit to continue")
        st.stop()

    # Display results and data after submit
    else:
        st.write("You entered:", geo_id)
        st.write(
            "\n<h1 style='text-align: center;'>" + geo_id + "</h1>\n",
            unsafe_allow_html=True,
        )
        if not st.session_state.processed:
            with st.spinner("Processing..."):
                run_process(geo_id)
        exp_data = st.session_state.exp_data
        general_info = st.session_state.general_info
        clin_data = st.session_state.clin_data
        warn = st.session_state.warn
        data_extra = st.session_state.extra

        # Display unused files with select radio
        handle_extra(data_extra, exp_data, clin_data)

        # Display data and selection of clinical data
        form, clin_data = display_data(exp_data, general_info, clin_data)
        display_column_selection(form, clin_data, warn)

    # Submit button for exporting files
    if "show_secondary" not in st.session_state:
        st.session_state.show_secondary = False

    # Response to export button click when no existing file found
    if not st.session_state.show_secondary:
        st.button(
            "Export files",
            on_click=on_main_click,
            args=(exp_data, general_info, clin_data, geo_id),
        )

    # Response to export button click when existing file found
    if st.session_state.show_secondary:
        st.warning(
            "Result files are found in the folder, overwrite existing files?\n"
            "Remember to close files before overwriting"
        )
        col1, col2 = st.columns(2)
        # Option Yes
        with col1:
            st.button(
                "Yes",
                on_click=on_yes_click,
                args=(exp_data, general_info, clin_data, geo_id),
            )
        # Option No
        with col2:
            st.button("No", on_click=on_no_click)


# def plotting():

#     # display datasets found
#     dataset_list = get_available_datasets()
#     geo_id, submit_button_geo = submit_geo_id(dataset_list)

#     # continue only if submit button was clicked
#     if submit_button_geo:
#         st.session_state.button_GEO = "clicked"
#         st.session_state.processed = False
#         st.session_state.button_extra = False
#     if "button_GEO" not in st.session_state:
#         st.write("Click Submit to continue")
#         # st.stop()

#     # results and data after submit
#     else:
#         st.write("You entered:", geo_id)
#         st.write(
#             "\n<h1 style='text-align: center;'>" + geo_id + "</h1>\n",
#             unsafe_allow_html=True,
#         )