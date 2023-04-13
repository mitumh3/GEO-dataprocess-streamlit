import os

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from plot_utils import *
from scipy.stats import zscore

load_dotenv()
RESULT_PATH = os.getenv("result_path")
geo_id = "GSE157103"
st.set_option("deprecation.showPyplotGlobalUse", False)
pd.options.mode.chained_assignment = None

# Load data from .csv file in processed_data
clinical_data = pd.read_csv(f"{RESULT_PATH}/{geo_id}/{geo_id}_clinical.csv")
expression_data = pd.read_csv(f"{RESULT_PATH}/{geo_id}/{geo_id}_expression.csv")


# Generate data for plotting
data = PlotData(clinical_data, expression_data)
data.generate_data()


# # - Line chart:
def line_chart(data):
    data = pd.read_csv("<filename.csv>")

    chart = (
        alt.Chart(data).mark_line().encode(x="x_variable", y="y_variable").interactive()
    )

    st.write(chart)


# - Scatter plot:
def scatter_plot(data):
    chart = (
        alt.Chart(data)
        .mark_circle()
        .encode(
            x=alt.X("index"), y=alt.Y("A1BG"), color=alt.Color("binary_disease state")
        )
        .interactive()
    )

    st.write(chart)


# - Bar chart:
def bar_chart(data):
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(x="A1BG", y="count()", color="binary_disease state")
        .interactive()
    )

    st.write(chart)


def clustermap(data: DataFrame, label=None):
    st.header("Expression Heatmap for Selected Genes")
    label = label.squeeze()
    unique_class = label.unique()
    colors = sns.light_palette("green", len(unique_class))

    lut = dict(zip(unique_class, colors))
    row_colors = label.map(lut)
    fig, axes = plt.subplots(1, 1)
    sns_heatmap = sns.clustermap(
        data,
        # cmap="Blues",
        cmap="seismic",
        col_cluster=False,
        metric="euclidean",
        method="complete",
        cbar_kws={"label": "Expression"},
        linewidths=0.2,
        col_colors=row_colors,
        cbar_pos=(0, 0.2, 0.03, 0.4),
    )
    ax = sns_heatmap.ax_heatmap
    ax.set_ylabel("")
    plt.setp(sns_heatmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    st.pyplot()


df = data.get_expression(num_rows=10000, normalize=True)
df
# label = data.binary_data
label = data.get_label("disease state")
label
clustermap(df, label)
