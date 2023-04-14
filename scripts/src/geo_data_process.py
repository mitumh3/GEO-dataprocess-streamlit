import streamlit as st
from display_helper import *
from main_helper import *
from streamlit import session_state as cache

# from plotting import *


def geo_data_process():
    """
    Main cache:
    - button_GEO: summit button of accesion ID form
    - geo_id: accession ID input
    - button_extra: submit button of unused data
    - data_exp: expression data
    - data_clin: clinical data
    - data_extra: unused data
    """

    st.write("# Process Data 👋")
    cache.button_extra = False

    # display datasets found
    data_lst = get_available_datasets()
    geo_id, submit_button_geo = submit_geo_id(data_lst)

    # continue only if submit button was clicked
    if submit_button_geo:
        cache.button_GEO = "clicked"
        cache.processed = False
        cache.button_extra = False
    if "button_GEO" not in cache:
        st.write("Click Submit to continue")
        st.stop()

    # Display results and data after submit
    else:
        st.write("You entered:", geo_id)
        st.write(
            "\n<h1 style='text-align: center;'>" + geo_id + "</h1>\n",
            unsafe_allow_html=True,
        )
        if not cache.processed:
            with st.spinner("Processing..."):
                run_process(geo_id)
        data_exp = cache.data_exp
        data_general_info = cache.data_general_info
        data_clin = cache.data_clin
        warn = cache.warn
        data_extra = cache.data_extra

        # Display unused files with select radio
        handle_extra(data_extra, data_exp, data_clin)

        # Display data and selection of clinical data
        form, data_clin = display_data(data_exp, data_general_info, data_clin)
        display_column_selection(form, data_clin, warn)

    # Submit button for exporting files
    if "show_secondary" not in cache:
        cache.show_secondary = False

    # Response to export button click when no existing file found
    if not cache.show_secondary:
        st.button(
            "Export files",
            on_click=on_main_click,
            args=(data_exp, data_general_info, data_clin, geo_id),
        )

    # Response to export button click when existing file found
    if cache.show_secondary:
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
                args=(data_exp, data_general_info, data_clin, geo_id),
            )
        # Option No
        with col2:
            st.button("No", on_click=on_no_click)
    st.write(cache)


# def plotting():

#     # display datasets found
#     dataset_list = get_available_datasets()
#     geo_id, submit_button_geo = submit_geo_id(dataset_list)

#     # continue only if submit button was clicked
#     if submit_button_geo:
#         cache.button_GEO = "clicked"
#         cache.processed = False
#         cache.button_extra = False
#     if "button_GEO" not in cache:
#         st.write("Click Submit to continue")
#         # st.stop()

#     # results and data after submit
#     else:
#         st.write("You entered:", geo_id)
#         st.write(
#             "\n<h1 style='text-align: center;'>" + geo_id + "</h1>\n",
#             unsafe_allow_html=True,
#         )
