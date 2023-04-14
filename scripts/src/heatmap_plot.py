import os

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

    feature_selection = st.selectbox("***Choose datasets***", dataset_lst)
    cache.geo_id = feature_selection

    # Stop if no geo_id found and Delete pre-option if geo_id found
    if cache.geo_id == "":
        st.write("Summit your dataset choice to continue")
        st.stop()
    elif dataset_lst[0] == "":
        st.experimental_rerun()

    ## OPTIONS OF PLOT TYPES
    # Create radio select
    options = ["Heatmap", "Bar Plot", "ROC"]
    st.radio(
        label="***Select plot type***",
        options=options,
        key="plot_option",
        horizontal=True,
    )
    # st.write(cache)

    # Stop if no plot type chosen
    if "plot_option" not in cache:
        st.stop()

    ## PARAMETERS OF THE CHOSEN PLOT
