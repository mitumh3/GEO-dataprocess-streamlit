import json
import os
from dataclasses import dataclass
from itertools import count

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as cache

from .data_utils import sort_by_label
from .plotting import (
    bar_chart,
    cluster,
    dendrogram,
    heatmap_func,
    pca_2d,
    pca_3d,
    plot_explained_variance,
)

load_dotenv()
FIGURE_PATH = os.getenv("figure_path")


@dataclass
class Graph:
    OPTIONS = None
    OPT = None
    GRAPH_DICT = None
    PARA_DICT = None
    INPUT = {}
    ROW_TAKE = None

    DF_Z = None
    DF_LABEL = None
    ID = None

    def __init__(self):
        with open("scripts/src/plot/graph_parameters.json", mode="r") as f:
            self.GRAPH_DICT = json.load(f)
        self.OPTIONS = self.GRAPH_DICT.keys()

    def draw_pca(self, data, dimension, white_background):
        df_exp = data.get_expression()
        patient_id = data.patient_id
        if not white_background:
            plot_bgcolor = "rgb(250, 250, 250)"
        else:
            plot_bgcolor = "white"

        if dimension == 2:
            fig = pca_2d(
                df_exp, cache.label, cache.label_title, patient_id, plot_bgcolor
            )
        elif dimension == 3:
            fig = pca_3d(
                df_exp, cache.label, cache.label_title, patient_id, plot_bgcolor
            )
        else:
            fig = plot_explained_variance(df_exp, plot_bgcolor)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        return fig

    def set_para(self):
        self.OPT = cache.graph_opt
        self.PARA_DICT = self.GRAPH_DICT[self.OPT]
        if self.OPT == "Heatmap":
            self.PARA_DICT["advanced"]["label_title"]["value"] = ", ".join(
                cache.label_title
            )

    def display_select_para(self, para_name, value, description, col):
        with col:
            selection = st.selectbox(
                f"***{description}***",
                value,
                key=f"{self.OPT}_{para_name}",
            )
        return selection

    def display_input_para(self, para_name, value, description, col):
        with col:
            selection = st.text_input(
                f"***{description}***",
                value,
                key=f"{self.OPT}_{para_name}",
            )
        return selection

    def display_check_para(self, para_name, value, description, col):
        with col:
            if value == "False":
                value = False
            else:
                value = True
            selection = st.checkbox(
                f"***{description}***",
                value,
                key=f"{self.OPT}_{para_name}",
            )
        return selection

    def display_slide_para(self, para_name, value, description, col):
        with col:
            selection = st.slider(
                f"***{description}***",
                min_value=value[0],
                max_value=value[2],
                value=value[1],
                step=value[3],
                key=f"{self.OPT}_{para_name}",
            )
        return selection

    def display_cpicker_para(self, para_name, value, description, col):
        with col:
            selection = st.color_picker(
                f"***{description}***",
                value,
                key=f"{self.OPT}_{para_name}",
            )
        return selection

    def display_para(self, level):
        para_level_dict = self.PARA_DICT[level]

        max_row = int(max(para_level_dict.values(), key=lambda x: int(x["row"]))["row"])
        for row_idx in range(max_row):
            para_col_lst = [
                para_name
                for para_name, properties in para_level_dict.items()
                if int(properties["row"]) - 1 == row_idx
            ]
            cols = st.columns(len(para_col_lst))

            for para_name, col in zip(para_col_lst, cols):
                para_type = para_level_dict[para_name]["type"]
                value = para_level_dict[para_name]["value"]
                row = para_level_dict[para_name]["row"]
                description = para_level_dict[para_name]["description"]

                if para_type == "select":
                    selection = self.display_select_para(
                        para_name, value, description, col
                    )
                elif para_type == "input":
                    selection = self.display_input_para(
                        para_name, value, description, col
                    )
                elif para_type == "check":
                    selection = self.display_check_para(
                        para_name, value, description, col
                    )
                elif para_type == "slide":
                    selection = self.display_slide_para(
                        para_name, value, description, col
                    )
                elif para_type == "cpicker":
                    selection = self.display_cpicker_para(
                        para_name, value, description, col
                    )
                elif para_type == "code":
                    selection = self.display_input_para(
                        para_name, value, description, col
                    )

                selection = self.check_input_para(selection)
                cache[self.OPT][para_name] = selection

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

    def load_heatmap_paras(self, label_cluster, row_cluster, col_cluster):
        if col_cluster:
            col_clusters, col_linkage_matrix = cluster(
                self.DF_Z.T,
                self.INPUT["distance_metric"],
                self.INPUT["link_method"],
                self.INPUT["col_cluster_threshold"],
            )
        else:
            col_clusters = []

        # TODO: create a single cluster that control label and col

        # Cluster by labels
        if label_cluster:
            self.DF_Z, self.DF_LABEL, self.ID = sort_by_label(
                self.DF_Z, self.DF_LABEL, self.ID, col_clusters
            )

        # Cluster by rows
        if row_cluster:
            row_clusters, row_linkage_matrix = cluster(
                self.DF_Z,
                self.INPUT["distance_metric"],
                self.INPUT["link_method"],
                self.INPUT["row_cluster_threshold"],
            )
            # Use the cluster assignments to reorder the rows of the dataframe
            self.DF_Z = self.DF_Z.iloc[np.argsort(row_clusters)]

            if self.INPUT["dendrogram"]:
                dendrogram_fig = dendrogram(
                    row_linkage_matrix, self.INPUT["row_cluster_threshold"]
                )
                cache.dendrogram = dendrogram_fig

    def load_input_and_data(self, data):
        self.INPUT = cache[self.OPT]
        if cache.run == "test":
            self.ROW_TAKE = int(cache.num_rows)
        elif cache.run == "target":
            self.ROW_TAKE = cache.target
        else:
            self.ROW_TAKE = -1

        # Get num_rows and scale for getting data
        if "scale" in self.INPUT:
            scale = self.INPUT["scale"]
        if "label_sort" in self.INPUT:
            label_cluster = self.INPUT["label_sort"]
        if "row_cluster" in self.INPUT:
            row_cluster = self.INPUT["row_cluster"]
        if "col_cluster" in self.INPUT:
            col_cluster = self.INPUT["col_cluster"]

        # Get data
        self.DF_Z = data.get_expression(row_take=self.ROW_TAKE, scaler=scale)
        self.DF_LABEL = cache.label
        self.ID = data.patient_id

        if self.OPT == "Heatmap":
            self.load_heatmap_paras(label_cluster, row_cluster, col_cluster)

        self.ID = list(self.ID.squeeze())

        # Caching input
        cache.input = {
            "parameters": self.INPUT,
            "row_take": self.ROW_TAKE,
            "df_label": self.DF_LABEL,
            "id": self.ID,
        }

        cache.df_z = self.DF_Z

    def display_para_options(self, data):
        # Create form
        with st.form("graph_para"):
            try:
                self.display_para("simple")
            except Exception as e:
                print(f"Extracting simple paras failed: {e}")
            with st.expander("Advanced settings"):
                try:
                    self.display_para("advanced")
                except Exception as e:
                    print(f"Extracting advanced paras failed: {e}")
            submit_button = st.form_submit_button("Submit", type="primary")

        if submit_button:
            # Load input parameters
            self.load_input_and_data(data)


def display_run_type(gene_lst):
    ## TEST OR FULL RUN
    st.radio(
        label="***Select run type:***",
        options=[
            "test",
            "full",
            "target",
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
    elif cache.run == "target":
        st.multiselect(
            label="Target",
            label_visibility="collapsed",
            options=gene_lst,
            key="target",
        )


# @st.cache_data
def draw_graph(_df_z, input_dict):
    # Draw plot
    if cache.graph_opt == "Heatmap":
        fig = heatmap_func(
            input_dict["parameters"], _df_z, input_dict["df_label"], input_dict["id"]
        )
        # Display plot
        st.plotly_chart(fig, use_container_width=True, theme=None)
    elif cache.graph_opt == "Bar Plot":
        # st.write(input_dict)
        fig = bar_chart(
            input_dict["parameters"], _df_z, input_dict["df_label"], input_dict["id"]
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
    return fig


def disply_save_button(fig, key, name):
    # Check existence of figure folder
    if not os.path.exists(FIGURE_PATH):
        os.mkdir(FIGURE_PATH)
    file_path = f"{FIGURE_PATH}/{cache.geo_id}"
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    col1, col2 = st.columns([4, 1])
    with col1:
        file_name = st.text_input(
            "file name",
            value=name,
            placeholder="Input file name",
            label_visibility="collapsed",
        )
    with col2:
        save_button = st.button(
            "Save as pdf", key=key, use_container_width=True, type="primary"
        )
    if save_button:
        fig.write_image(f"{file_path}/{file_name}.pdf", format="pdf")
