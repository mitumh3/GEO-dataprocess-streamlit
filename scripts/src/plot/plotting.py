import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# # - Line chart:
import plotly.io as pio
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

pio.templates.default = "seaborn"
pio.templates["seaborn"].layout.font.family = "Roboto, sans-serif"


def pca_2d(df_exp, df_label, label_title, patient_id):
    # Extract the numerical features from the data
    data = df_exp.T

    # Perform PCA with 2 components
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    total_var = pca.explained_variance_ratio_.sum() * 100

    # Create a new dataframe with the principal components
    df_pca = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

    # Add the corresponding labels to the data points
    df_pca["id"] = patient_id.T["index"]
    df_pca[label_title] = df_label.T[label_title]
    try:
        df_pca[label_title] = df_pca[label_title].astype(str)
    except:
        pass
    # Plot the PCA results using Plotly

    fig = go.Figure(
        px.scatter(
            df_pca,
            x="PC1",
            y="PC2",
            color=label_title,
            # symbol=label_title,
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.D3,
            marginal_x="box",  # display distribution marginal on x-axis
            marginal_y="violin",  # display distribution marginal on y-axis
            trendline="ols",  # display trendline, OLS regression
            hover_name="id",
        )
    )
    fig.update_traces(marker=dict(size=7), selector=dict(mode="markers"))
    fig.update_layout(
        height=700,
        width=1000,
        title=dict(
            text=f"Total Explained Variance: {total_var:.2f}%",
            font=dict(size=20, color="#31333F"),
            x=0.5,
            y=0.97,
            xanchor="center",
            yanchor="top",
        ),
        hovermode="closest",
        xaxis=dict(
            showgrid=True,
            title="Principal Component 1",
            titlefont=dict(
                size=14,
                color="#31333F",
            ),
        ),
        yaxis=dict(
            showgrid=True,
            title="Principal Component 2",
            titlefont=dict(
                size=14,
                color="#31333F",
            ),
        ),
        legend=dict(
            title=dict(
                text=label_title,
                font=dict(size=14, color="#31333F"),
            ),
        ),
    )

    fig.update_xaxes(
        gridwidth=1,
        zerolinewidth=1,
        zerolinecolor="rgba(180, 180, 180, 0.1)",
    )
    fig.update_layout(plot_bgcolor="rgb(250, 250, 250)")
    return fig


def pca_3d(df_exp, df_label, label_title, patient_id):
    pio.templates.default = "plotly_white"
    pio.templates["plotly_white"].layout.font.family = "Roboto, sans-serif"

    # Extract the numerical features from the data
    data = df_exp.T

    # Perform PCA with 2 components
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(data)
    total_var = pca.explained_variance_ratio_.sum() * 100

    # Create a new dataframe with the principal components
    df_pca = pd.DataFrame(data=principal_components, columns=["PC1", "PC2", "PC3"])

    # Add the corresponding labels to the data points
    df_pca["id"] = patient_id.T["index"]
    df_pca[label_title] = df_label.T[label_title]
    try:
        df_pca[label_title] = df_pca[label_title].astype(str)
    except:
        pass

    fig = px.scatter_3d(
        df_pca,
        x="PC1",
        y="PC2",
        z="PC3",
        color=label_title,
        opacity=0.8,
        # symbol=label_title,
        color_discrete_sequence=px.colors.qualitative.D3,
        hover_name="id",
    )
    fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))
    fig.update_layout(
        height=700,
        width=1000,
        title=dict(
            text=f"Total Explained Variance: {total_var:.2f}%",
            font=dict(size=20, color="#31333F"),
            x=0.5,
            y=0.97,
            xanchor="center",
            yanchor="top",
        ),
        hovermode="closest",
        legend=dict(
            title=dict(
                text=label_title,
                font=dict(size=14, color="#31333F"),
            ),
        ),
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgb(250, 250, 250)",
                showbackground=True,
                title="Principal Component 1",
                titlefont=dict(
                    size=14,
                    color="#31333F",
                ),
            ),
            yaxis=dict(
                backgroundcolor="rgb(250, 250, 250)",
                showbackground=True,
                title="Principal Component 2",
                titlefont=dict(
                    size=14,
                    color="#31333F",
                ),
            ),
            zaxis=dict(
                backgroundcolor="rgb(250, 250, 250)",
                showbackground=True,
                title="Principal Component 3",
                titlefont=dict(
                    size=14,
                    color="#31333F",
                ),
            ),
            # bgcolor="cyan",
        ),
        plot_bgcolor="rgb(12,163,135)",
    )
    return fig


def plot_explained_variance(df_exp):
    pca = PCA()
    pca.fit(df_exp)
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    fig = go.Figure(
        px.area(
            x=range(1, exp_var_cumul.shape[0] + 1),
            y=exp_var_cumul,
        )
    )
    fig.update_layout(
        height=700,
        width=1000,
        title=dict(
            text=f"Total Explained Variance",
            font=dict(size=20, color="#31333F"),
            x=0.5,
            y=0.97,
            xanchor="center",
            yanchor="top",
        ),
        hovermode="closest",
        xaxis=dict(
            showgrid=True,
            title="Number of Components",
            titlefont=dict(
                size=14,
                color="#31333F",
            ),
        ),
        yaxis=dict(
            showgrid=True,
            title="Explained Variance",
            titlefont=dict(
                size=14,
                color="#31333F",
            ),
        ),
        plot_bgcolor="rgb(250, 250, 250)",
    )
    return fig


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
        autosize=True,
        margin=dict(t=100, b=20, l=10, r=10),
        height=paras["plot_layout"]["Fig height"],
        width=paras["plot_layout"]["Fig width"],
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
    )

    return fig
