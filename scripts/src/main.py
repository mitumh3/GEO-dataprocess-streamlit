import data_process
import plot
import streamlit as st

# Format page
st.set_page_config(page_title="Process Data", layout="wide")
if __name__ == "__main__":
    page_names_to_funcs = {"Process Data": data_process.layout, "Plot": plot.layout}

    demo_name = st.sidebar.selectbox("Choose tools", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
    # main()
st.container()

# TODO some expression file separated by comma (csv file)
