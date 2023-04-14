import streamlit as st
from geo_data_process import geo_data_process
from heatmap_plot import heatmap_plot
from streamlit import session_state as cache

# Format page
st.set_page_config(page_title="Process Data")
if __name__ == "__main__":
    page_names_to_funcs = {
        "—": geo_data_process,
        "Plotting Demo": heatmap_plot
        # "Mapping Demo": mapping_demo,
        # "DataFrame Demo": data_frame_demo
    }

    demo_name = st.sidebar.selectbox("Choose tools", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
    # main()
