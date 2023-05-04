import pandas as pd
import streamlit as st
from main_helper import RESULT_PATH, get_processed_dataset, start_page
from streamlit import session_state as cache

from .data_utils import PlotData
from .plot_helper import Graph, display_run_type, disply_save_button, draw_graph
from .plotting import bar_chart

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

    # SELECT LABEL:
    # st.subheader("Select label")
    label_lst = [x for x in data.labels.index]
    label_title = st.multiselect("***Select label:***", label_lst)

    new_label_title = [old_label.split("_", maxsplit=1)[1] for old_label in label_title]
    cache.label = data.get_label(label_title, new_label_title)
    cache.label_title = new_label_title

    tab1, tab2, tab3 = st.tabs(["Plot", "Preview data", "PCA"])

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

        display_run_type(data.gene_lst)
        st.write(
            f"\n<h1 style='text-align: center;'> {cache.geo_id} - {cache.graph_opt} </h1>\n",
            unsafe_allow_html=True,
        )

    with tab3:
        if not cache.label_title:
            st.info("Choose labels to continue")
        else:
            if len(cache.label_title) > 1:
                st.warning(
                    "For the best results, it's recommended to analyze PCA with only one label. Any additional labels beyond the first will be ignored."
                )
            col1pca, col2pca = st.columns(2, gap="large")
            dimension = col1pca.select_slider(
                "***Choose dimension number:***", options=[2, 3, "all"]
            )
            col2pca.write("***White background***")
            white_background = col2pca.checkbox(
                "White background", key="pca_bg", label_visibility="collapsed"
            )
            with st.expander("***Result:***", expanded=True):
                pca_output = graph.draw_pca(data, dimension, white_background)

            sulfix = "-".join(cache.label_title)
            disply_save_button(pca_output, "save_pca", f"pca{dimension}d_{sulfix}")

    with tab1:
        graph.set_para()

        ## PARAMETERS OF THE CHOSEN PLOT
        # Load paras
        st.subheader("Set parameters:")

        graph.display_para_options(data)

        if "input" not in cache:
            st.stop()

        if cache.graph_opt != "Heatmap":
            pass
        elif cache.input["parameters"]["dendrogram"]:
            with st.expander("***Dendrogram:***", expanded=True):
                with st.spinner("Drawing dendrogram..."):
                    st.pyplot(cache.dendrogram)

        with st.expander("***Result:***", expanded=True):
            with st.spinner("Drawing graph..."):
                # Draw
                output = draw_graph(cache.df_z, cache.input)
        sulfix = "-".join(cache.label_title)
        disply_save_button(output, f"save_{graph.OPT}", f"{graph.OPT}_{sulfix}")
