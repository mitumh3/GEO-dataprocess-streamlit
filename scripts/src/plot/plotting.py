import altair as alt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots


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


def map_label_colors(label, label_color):
    label = label.squeeze()
    unique_class = label.unique()
    colors = sns.light_palette(label_color, len(unique_class))

    lut = dict(zip(unique_class, colors))
    target_colors = label.map(lut)
    return target_colors


def heatmap_func(input_paras, df, df_label, patient_id):
    """ """
    # Define the input parameters for the cluster map
    paras = input_paras

    label_height = paras["label_layout"]["height"]
    x_title = paras["xaxis"]["X-axis title"]
    y_title = paras["yaxis"]["Y-axis title"]
    cbar_title = paras["cbar_layout"]["Cbar title"]

    # Define cols and rows of figure
    fig = make_subplots(
        rows=2,
        cols=1,
        column_widths=[0.9],
        row_heights=[label_height, 1 - label_height],
        vertical_spacing=paras["label_layout"]["spacing"],
    )

    # Create a bar for labeling
    old_label_name = df_label.index.values[0]
    df_label.rename(
        index={old_label_name: paras["label_layout"]["Label title"]}, inplace=True
    )
    label_bar = go.Heatmap(
        z=df_label,
        x=patient_id,
        y=df_label.index,
        colorscale=paras["para"]["Label pallete"],
        showscale=False,
        hovertemplate=f"Label: %{{z}}<br>{x_title}: %{{x}}",
        name="",
    )

    # Create heatmap
    heatmap = go.Heatmap(
        z=df,
        x=patient_id,
        y=df.index,
        colorscale=paras["para"]["Graph pallete"],
        hoverongaps=False,
        hovertemplate=f"{cbar_title}: %{{z}}<br>{y_title}: %{{y}}<br>{x_title}: %{{x}}",
        zmid=paras["para"]["Mid point"],
        colorbar=dict(
            thickness=paras["cbar_pos"]["Cbar width"],
            len=paras["cbar_pos"]["Cbar height"],
            title=cbar_title,
            titlefont=dict(
                family=paras["cbar_layout"]["Font"],
                size=paras["cbar_layout"]["Size"],
                color=paras["cbar_layout"]["color"],
            ),
            tickfont=dict(
                family=paras["cbar_layout"]["tickfont"],
                size=paras["cbar_layout"]["tickfontsize"],
            ),
            xpad=paras["cbar_pos"][
                "Cbar right"
            ],  # increase gap between heatmap and colorbar horizontally
            outlinewidth=paras["cbar_pos"]["outlinewidth"],
        ),
        name="",
    )

    # Add label bar and heatmap to figure
    fig.append_trace(label_bar, row=1, col=1)
    fig.append_trace(heatmap, row=2, col=1)

    # # Update figure layout
    fig.update_layout(
        title=paras["plot_layout"]["Plot title"],
        xaxis2=dict(
            title=x_title,
            titlefont=dict(
                family=paras["xaxis"]["Font"],
                size=paras["xaxis"]["Size"],
                color=paras["xaxis"]["color"],
            ),
            tickfont=dict(
                family=paras["xaxis"]["tickfont"],
                size=paras["xaxis"]["tickfontsize"],
            ),
            tickangle=paras["xaxis"]["X-axis tick angle"],
        ),
        yaxis2=dict(
            title=y_title,
            titlefont=dict(
                family=paras["yaxis"]["Font"],
                size=paras["yaxis"]["Size"],
                color=paras["yaxis"]["color"],
            ),
            tickfont=dict(
                family=paras["yaxis"]["tickfont"],
                size=paras["yaxis"]["tickfontsize"],
            ),
            autorange="reversed",
        ),
        xaxis1=dict(
            showticklabels=False,
            ticks="",
        ),
        yaxis1=dict(
            tickfont=dict(
                family=paras["label_layout"]["Font"],
                size=paras["label_layout"]["Size"],
            ),
        ),
        autosize=True,
        margin=dict(t=100, b=20, l=10, r=10),
        height=paras["plot_layout"]["Fig height"],
        width=paras["plot_layout"]["Fig width"],
    )

    return fig
