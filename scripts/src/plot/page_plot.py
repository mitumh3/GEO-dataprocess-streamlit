import json
import os
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from main_helper import get_available_datasets, start_page
from streamlit import session_state as cache

from .plot_utils import RESULT_PATH, PlotData
from .plotting import heatmap_func

st.set_option("deprecation.showPyplotGlobalUse", False)


# cache
# display datasets found
def layout():
    """
    Main cache:
    - geo_id


    """
    start_page("heatmap", ["geo_id"])
    ## FORM OF CHOOSING DATASETS
    # Create form
    dataset_lst = get_available_datasets()
    if cache.geo_id == "":
        dataset_lst.insert(0, "")
    else:
        dataset_lst.remove(cache.geo_id)
        dataset_lst.insert(0, cache.geo_id)

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
        data_clin = pd.read_csv(f"{RESULT_PATH}/{geo_id}/{geo_id}_clinical.csv")
        data_exp = pd.read_csv(f"{RESULT_PATH}/{geo_id}/{geo_id}_expression.csv")
        data = PlotData(data_clin, data_exp)
        data.generate_data()
        cache.data = data
        cache.processed = geo_id

    # Load data for plotting
    data = cache.data

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
    graph.set_para()

    ## TEST OR FULL RUN
    st.radio(
        label="***Select run:***",
        options=["full", "test"],
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
    label_lst = [x.split("_", maxsplit=1)[1] for x in data.binary_data.index]
    label = st.selectbox("Choose label", label_lst, label_visibility="collapsed")
    cache.label = data.get_label(label)

    ## PARAMETERS OF THE CHOSEN PLOT
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


"""
PROBLEM:
Untick the checkbox does not return advanced settings to default
"""


@dataclass
class Graph:
    OPTIONS = ["Heatmap", "Bar Plot", "ROC"]
    OPT = None
    GRAPH_DICT = None
    PARA_DICT = None
    INPUT = {}

    def set_para(self):
        self.OPT = cache.graph_opt
        with open("scripts/src/graph_parameters.json", mode="r") as f:
            self.GRAPH_DICT = json.load(f)
        self.PARA_DICT = self.GRAPH_DICT[self.OPT]

    def display_select_para(self, paras, sulfix):
        columns = st.columns(len(paras))
        for col, para_name in zip(columns, paras):
            with col:
                para_opts = paras[para_name]
                selection = st.selectbox(
                    f"***{para_name}***",
                    para_opts,
                    key=f"{self.OPT}_{sulfix}_{para_name}",
                )
            selection = self.check_input_para(selection)
            cache[self.OPT][sulfix][para_name] = selection

    def display_input_para(self, paras, sulfix):
        columns = st.columns(len(paras))
        for col, para_name in zip(columns, paras):
            with col:
                para_opts = paras[para_name]
                selection = st.text_input(
                    f"***{para_name}***",
                    para_opts,
                    key=f"{self.OPT}_{sulfix}_{para_name}",
                )
            selection = self.check_input_para(selection)
            cache[self.OPT][sulfix][para_name] = selection

    def display_check_para(self, paras, sulfix):
        columns = st.columns(len(paras))
        for col, para_name in zip(columns, paras):
            with col:
                para_opts = paras[para_name]
                selection = st.checkbox(
                    f"***{para_name}***",
                    para_opts,
                    key=f"{self.OPT}_{sulfix}_{para_name}",
                )
            selection = self.check_input_para(selection)
            cache[self.OPT][sulfix][para_name] = selection

    def display_para(self, level):
        # try:
        para_level_dict = self.PARA_DICT[level]
        for key, paras in para_level_dict.items():
            para_type, sulfix = key.split("_", maxsplit=1)
            cache[self.OPT][sulfix] = {}
            if para_type == "select":
                self.display_select_para(paras, sulfix)
            elif para_type == "input":
                self.display_input_para(paras, sulfix)
            elif para_type == "check":
                self.display_check_para(paras, sulfix)
        # except:
        #     pass

    def check_input_para(self, value):
        if isinstance(value, list):
            value = value[0]
        if value == "":
            value = None
        if not isinstance(value, bool):
            try:
                value = float(value)
            except:
                pass
        return value

    def load_input_paras(self):
        self.INPUT = cache[self.OPT]
        if cache.run == "test":
            self.INPUT["num_rows"] = int(cache.num_rows)
        else:
            self.INPUT["num_rows"] = -1

        self.INPUT["label"] = cache.label

        return self.INPUT

    def draw_graph(self, data):
        num_rows = self.INPUT.pop("num_rows")
        for key, value in self.INPUT.items():
            if "Scale" in value:
                scale = self.INPUT[key].pop("Scale")
        df = data.get_expression(num_rows=num_rows, scaler=scale)
        fig = heatmap_func(
            self.INPUT,
            df,
        )
        st.plotly_chart(fig, use_container_width=True)
