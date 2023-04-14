import os
from dataclasses import dataclass

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from display_helper import *
from dotenv import load_dotenv
from main_helper import *
from plot_utils import *
from streamlit import _SessionStateProxy
from streamlit import session_state as cache

st.set_option("deprecation.showPyplotGlobalUse", False)
load_dotenv()
RESULT_PATH = os.getenv("result_path")


# cache
# display datasets found
def heatmap_plot():
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

    st.selectbox("***Choose dataset:***", dataset_lst, key="geo_id")

    # Stop if no geo_id found and Delete pre-option if geo_id found
    if cache.geo_id == "":
        st.write("Summit your dataset choice to continue")
        st.stop()
    elif dataset_lst[0] == "":
        st.experimental_rerun()

    # Plot
    graph = Graph()

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

    st.write(
        f"\n<h1 style='text-align: center;'> {cache.geo_id} - {cache.graph_opt} </h1>\n",
        unsafe_allow_html=True,
    )

    ## PARAMETERS OF THE CHOSEN PLOT
    # Create form
    st.write("***Set parameters:***")
    with st.form("graph_para"):
        graph.set_para()
        graph.display_select_para("simple select")
        if "adv_set" not in cache:
            cache.adv_set = False
        if cache.adv_set == True:
            graph.display_select_para("advanced select")
        submitted = st.form_submit_button("Submit")
    st.checkbox("Advanced settings", key="adv_set")
    st.write(cache)


"""
PROBLEM:
Untick the checkbox does not return advanced settings to default
"""


@dataclass
class Graph:
    OPTIONS = ["Heatmap", "Bar Plot", "ROC"]
    OPT = None
    PARA = None
    PARA_DICT = {
        "Heatmap": {
            "simple select": {
                "Graph pallete": [
                    "Blues",
                    "seismic",
                    "coolwarm",
                    "plasma",
                    "other",
                ],
                "Label pallete": ["green", "other"],
                "Label": ["row", "column"],
            },
            "advanced select": {
                "Metric": ["euclidean"],
                "Method": ["complete", "average"],
                "Scale": ["Standard", "Min Max"],
            },
            "advanced input": {
                "CBAR_POS": [],
                "LINEWIDTH": [],
                "FIG_SIZE": [],
                "CBAR_KWS": [],
            },
        }
    }

    def set_para(self):
        self.OPT = cache.graph_opt
        self.PARA = self.PARA_DICT[self.OPT]

    def display_select_para(self, para_type):
        paras = self.PARA[para_type]

        columns = st.columns(len(paras))
        for col, para in zip(columns, paras):
            with col:
                para_opts = paras[para]
                selected_para = st.selectbox(para, para_opts, key=f"{self.OPT}_{para}")

    def adv_select(self):
        pass

    def adv_input(self):
        pass

    def display_advanced_para(self):
        paras = self.PARA["advanced select"]

        columns = st.columns(len(paras))
        for col, para in zip(columns, paras):
            with col:
                para_opts = para[para]
                selected_para = st.selectbox(para, para_opts, key=f"{self.OPT}_{para}")
