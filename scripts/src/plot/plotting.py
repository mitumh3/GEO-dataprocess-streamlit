from multiprocessing import Pool, cpu_count

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# # - Line chart:
import plotly.io as pio
import scipy.cluster.hierarchy as sch
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

pio.templates.default = "seaborn"
pio.templates["seaborn"].layout.font.family = "Roboto, sans-serif"
import sys

sys.setrecursionlimit(1000000)


def pca_2d(df_exp, df_label, label_title, patient_id, plot_bgcolor):
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
        df_pca[label_title] = df_pca[label_title].astype("category")
    except:
        pass
    # Plot the PCA results using Plotly

    fig = go.Figure(
        px.scatter(
            df_pca,
            x="PC1",
            y="PC2",
            color=label_title[0],
            # symbol=label_title[1],
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
                text=label_title[0],
                font=dict(size=14, color="#31333F"),
            ),
        ),
    )

    fig.update_xaxes(
        gridwidth=1,
        zerolinewidth=1,
        zerolinecolor="rgba(180, 180, 180, 0.1)",
    )
    fig.update_layout(plot_bgcolor=plot_bgcolor)
    return fig


def pca_3d(df_exp, df_label, label_title, patient_id, plot_bgcolor):
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
        df_pca[label_title] = df_pca[label_title].astype("catergory")
    except:
        pass

    fig = px.scatter_3d(
        df_pca,
        x="PC1",
        y="PC2",
        z="PC3",
        color=label_title[0],
        opacity=0.8,
        # symbol=label_title[1],
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
                text=label_title[0],
                font=dict(size=14, color="#31333F"),
            ),
        ),
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor=plot_bgcolor,
                showbackground=True,
                title="Principal Component 1",
                titlefont=dict(
                    size=14,
                    color="#31333F",
                ),
            ),
            yaxis=dict(
                backgroundcolor=plot_bgcolor,
                showbackground=True,
                title="Principal Component 2",
                titlefont=dict(
                    size=14,
                    color="#31333F",
                ),
            ),
            zaxis=dict(
                backgroundcolor=plot_bgcolor,
                showbackground=True,
                title="Principal Component 3",
                titlefont=dict(
                    size=14,
                    color="#31333F",
                ),
            ),
            # bgcolor="cyan",
        ),
    )
    return fig


def plot_explained_variance(df_exp, plot_bgcolor):
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
        plot_bgcolor=plot_bgcolor,
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


def custom_code(cmd):
    if cmd != None:
        eval(cmd)


def kmean_cluster(df):
    # Define the range of n_clusters to try
    range_n_clusters = range(1, 100)

    # Calculate the within-cluster sum of squares (WCSS) for different values of n_clusters
    wcss = []
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(df.transpose())
        wcss.append(kmeans.inertia_)

    # Create a plotly figure for the Elbow Method
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[i for i in range_n_clusters],
            y=wcss,
            mode="lines+markers",
            marker=dict(size=7),
            name="Cluster",
        )
    )

    fig.update_layout(
        title="Elbow Method for Optimal n_clusters",
        xaxis_title="Number of clusters",
        yaxis_title="Within-cluster sum of squares (WCSS)",
        template="plotly_white",
        width=800,
        height=500,
        font=dict(family="Calibri", size=14),
    )

    st.plotly_chart(fig)

    # Perform k-means clustering on the gene expression data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df)
    # Reorder the rows and columns of the DataFrame based on the clustering
    df_clustered = df.iloc[np.argsort(cluster_labels)]


def cluster(_df, metric, method, threshold):
    # Compute distance matrix and apply hierarchical clustering using average linkage
    distance_matrix = sch.distance.pdist(_df, metric=metric)
    linkage_matrix = sch.linkage(
        distance_matrix, method=method
    )  # you can change the linkage method here

    th = threshold  # choose a threshold value to cut the tree

    # Extract cluster assignments based on the dendrogram and threshold value
    clusters = sch.fcluster(linkage_matrix, th, criterion="distance")

    return clusters, linkage_matrix


def dendrogram(linkage_matrix, threshold):
    fig, ax = plt.subplots(figsize=(21, 7))
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("sample index")
    plt.ylabel("distance")
    sch.dendrogram(
        linkage_matrix,
        truncate_mode="mtica",
        color_threshold=threshold,  # set the threshold here
        above_threshold_color="gray",  # color the lines above the threshold gray
        orientation="top",
        leaf_rotation=45,
        leaf_font_size=8,
    )
    ax.axhline(y=threshold, color="r", linestyle="--")
    # plt.xticks(rotation=45)

    return fig


def cluster_worker(df, metric, method):
    # Compute distance matrix and apply hierarchical clustering using average linkage
    distance_matrix = sch.distance.pdist(df, metric=metric)
    linkage_matrix = sch.linkage(
        distance_matrix, method=method
    )  # you can change the linkage method here

    # Choose a threshold value to cut the tree
    th = 0.6  # choose a threshold value to cut the tree

    # Extract cluster assignments based on the dendrogram and threshold value
    clusters = sch.fcluster(linkage_matrix, th, criterion="distance")

    # Use the cluster assignments to reorder the rows of the dataframe
    genes_reordered = df.iloc[np.argsort(clusters)]
    return genes_reordered


def cluster_multi(df, n_jobs=None):
    if n_jobs is None:
        n_jobs = cpu_count()
    pool = Pool(processes=n_jobs)
    chunks = np.array_split(df, n_jobs)
    result = pd.concat(pool.map(cluster_worker, chunks))
    pool.close()
    pool.join()
    return result


def heatmap_func(input_paras, df, df_label, patient_id):
    """ """
    # Define the input parameters for the cluster map
    paras = input_paras

    label_height = paras["label_layout"]["height"]
    x_title = paras["xaxis"]["X-axis title"]
    y_title = paras["yaxis"]["Y-axis title"]
    cbar_title = paras["cbar_layout"]["Cbar title"]

    # Define cols and rows of figure
    num_rows = df_label.shape[0] + 1
    row_heights = [label_height for i in range(df_label.shape[0])]
    row_heights.append(1 - sum(row_heights))
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        column_widths=[0.9],
        row_heights=row_heights,
        vertical_spacing=paras["label_layout"]["spacing"],
        shared_xaxes=True,
    )

    # Create a bar for labeling
    new_name_lst = eval(paras["label_layout"]["Label title"])
    for old_name, new_name in zip(list(df_label.index.values), new_name_lst):
        df_label.rename(index={old_name: new_name}, inplace=True)
    for i, name in enumerate(reversed(df_label.index)):
        df_subset_label = df_label.loc[[name]]
        label_bar = go.Heatmap(
            z=df_subset_label,
            x=patient_id,
            y=df_subset_label.index,
            colorscale=paras["para"]["Label pallete"],
            showscale=False,
            hovertemplate=f"Label: %{{z}}<br>{x_title}: %{{x}}",
            name="",
        )
        fig.append_trace(label_bar, row=i + 1, col=1)

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
    # fig.append_trace(label_bar, row=1, col=1)
    fig.append_trace(heatmap, row=num_rows, col=1)

    # # Update figure layout
    fig.update_layout(
        title=paras["plot_layout"]["Plot title"],
        autosize=True,
        margin=dict(t=100, b=20, l=10, r=10),
        height=paras["plot_layout"]["Fig height"],
        width=paras["plot_layout"]["Fig width"],
    )

    # Main xaxis layout
    fig.layout[f"xaxis{num_rows}"].update(
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
        tickangle=paras["tick_angle"]["X-axis tick angle"],
    )

    # Main yaxis layout
    fig.layout[f"yaxis{num_rows}"].update(
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
        tickangle=paras["tick_angle"]["Y-axis tick angle"],
    )

    # Label y axes layout
    label_yaxes = [fig.layout[f"yaxis{e}"] for e in range(1, num_rows)]
    for yaxis in label_yaxes:
        yaxis.update(
            tickfont=dict(
                family=paras["label_layout"]["Font"],
                size=paras["label_layout"]["Size"],
                color=paras["label_layout"]["color"],
            ),
            tickangle=paras["tick_angle"]["Y-axis tick angle"],
        )
    custom_code(paras["custom"]["*Custom codes (For developers)"])
    return fig
