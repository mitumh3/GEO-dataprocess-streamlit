import streamlit as st
from geo_data_process import geo_data_process as geo

# Format page
st.set_page_config(page_title="Process Data")
st.session_state.initial = st.session_state

if __name__ == "__main__":
    page_names_to_funcs = {
        "—": geo,
        # "Plotting Demo": plotting
        # "Mapping Demo": mapping_demo,
        # "DataFrame Demo": data_frame_demo
    }

    demo_name = st.sidebar.selectbox("Choose tools", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
    # main()
