import pandas as pd
import streamlit as st
from main_helper import RESULT_PATH, get_processed_dataset, start_page
from streamlit import session_state as cache

from .data_utils import PlotData
from .plot_helper import Graph

st.set_option("deprecation.showPyplotGlobalUse", False)


# cache
# display datasets found
def layout():
    """
    Main cache:
    - geo_id


    """
    start_page("plot", ["geo_id"])

    ## FORM OF CHOOSING DATASETS
    # Create form
    dataset_lst = get_processed_dataset()
    try:
        if cache.geo_id == "":
            dataset_lst.insert(0, "")
        else:
            dataset_lst.remove(cache.geo_id)
            dataset_lst.insert(0, cache.geo_id)
    except:
        st.stop()

    col1_initial, col2_initial = st.columns(2)
    with col1_initial:
        st.selectbox("***Choose dataset:***", dataset_lst, key="geo_id")

    # Stop if no geo_id found and Delete blank option if geo_id chosen
    if cache.geo_id == "":
        st.write("Choose dataset to continue")
        st.stop()
    elif dataset_lst[0] == "":
        st.experimental_rerun()

    # Plot
    graph = Graph()

    # Plot data
    geo_id = cache.geo_id

    # Generate data and avoid repeated loading
    if "processed" not in cache:
        cache.processed = False
    if cache.processed != geo_id:
        with st.spinner(f"Loading dataset {geo_id}..."):
            data_clin = pd.read_csv(f"{RESULT_PATH}/{geo_id}/{geo_id}_clinical.csv")
            data_exp = pd.read_csv(f"{RESULT_PATH}/{geo_id}/{geo_id}_expression.csv")
            data = PlotData(data_clin, data_exp)
            data.generate_data()
            cache.data = data
            cache.processed = geo_id
    # Load data for plotting
    data = cache.data

    tab1, tab2 = st.tabs(["Plot", "Preview data"])

    with tab2:
        num_rows, num_cols = data.clinical_data.shape
        st.subheader("Processed clinical data")
        st.write(num_rows, " samples with ", num_cols, " features")
        st.write(data.clinical_data)

        st.subheader("Denote table")
        st.write(data.denote_log)

    with tab1:
        ## OPTIONS OF PLOT TYPES
        # Create radio select
        GRAPH_OPTS = graph.OPTIONS
        st.radio(
            label="***Select graph type:***",
            options=GRAPH_OPTS,
            key="graph_opt",
            horizontal=True,
        )
        # st.write(cache)

        # Stop if no plot type chosen
        if "graph_opt" not in cache:
            st.stop()
        cache[cache.graph_opt] = {}

        ## TEST OR FULL RUN
        st.radio(
            label="***Select run:***",
            options=[
                "test",
                "full",
            ],
            key="run",
            horizontal=True,
        )
        if cache.run == "test":
            st.text_input(
                label="Number of rows",
                label_visibility="collapsed",
                value="10",
                placeholder="Number of rows taken for test",
                key="num_rows",
            )

        st.write(
            f"\n<h1 style='text-align: center;'> {cache.geo_id} - {cache.graph_opt} </h1>\n",
            unsafe_allow_html=True,
        )

        # SELECT LABEL:
        st.subheader("Select label")
        label_lst = [x for x in data.labels.index]
        label_title = st.selectbox(
            "Choose label", label_lst, label_visibility="collapsed"
        )

        new_label_title = label_title.split("_", maxsplit=1)[1]
        cache.label_title = new_label_title
        cache.label = data.get_label(label_title, new_label_title)
        graph.set_para()

        ## PARAMETERS OF THE CHOSEN PLOT
        # Load paras
        st.subheader("Set parameters:")

        # Create form
        with st.form("graph_para"):
            graph.display_para("simple")

            with st.expander("Advanced settings"):
                graph.display_para("advanced")
            submit_button = st.form_submit_button("Submit", type="primary")

        if not submit_button:
            st.stop()

        with st.spinner("Drawing..."):
            # Load input parameters
            graph.load_input_paras()
            # Draw
            graph.draw_graph(data)
