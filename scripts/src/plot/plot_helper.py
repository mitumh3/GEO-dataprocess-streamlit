import json
from dataclasses import dataclass

import streamlit as st
from streamlit import session_state as cache

from .data_utils import sort_by_label
from .plotting import heatmap_func


@dataclass
class Graph:
    OPTIONS = None
    OPT = None
    GRAPH_DICT = None
    PARA_DICT = None
    INPUT = {}
    ROW_TAKE = None

    def __init__(self):
        with open("scripts/src/plot/graph_parameters.json", mode="r") as f:
            self.GRAPH_DICT = json.load(f)
        self.OPTIONS = self.GRAPH_DICT.keys()

    def set_para(self):
        self.OPT = cache.graph_opt
        self.PARA_DICT = self.GRAPH_DICT[self.OPT]
        if self.OPT == "Heatmap":
            self.PARA_DICT["advanced"]["input_label_layout"][
                "Label title"
            ] = cache.label_title

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
                if paras[para_name] == "False":
                    para_opts = False
                else:
                    para_opts = True
                selection = st.checkbox(
                    f"***{para_name}***",
                    value=para_opts,
                    key=f"{self.OPT}_{sulfix}_{para_name}",
                )
            selection = self.check_input_para(selection)
            cache[self.OPT][sulfix][para_name] = selection

    def display_slide_para(self, paras, sulfix):
        columns = st.columns(len(paras))
        for col, para_name in zip(columns, paras):
            with col:
                para_opts = paras[para_name]
                selection = st.slider(
                    f"***{para_name}***",
                    min_value=para_opts[0],
                    max_value=para_opts[2],
                    value=para_opts[1],
                    step=para_opts[3],
                    key=f"{self.OPT}_{sulfix}_{para_name}",
                )
            selection = self.check_input_para(selection)
            cache[self.OPT][sulfix][para_name] = selection

    def display_para(self, level):
        para_level_dict = self.PARA_DICT[level]
        for key, paras in para_level_dict.items():
            para_type, sulfix = key.split("_", maxsplit=1)
            if sulfix not in cache[self.OPT]:
                cache[self.OPT][sulfix] = {}
            if para_type == "select":
                self.display_select_para(paras, sulfix)
            elif para_type == "input":
                self.display_input_para(paras, sulfix)
            elif para_type == "check":
                self.display_check_para(paras, sulfix)
            elif para_type == "slide":
                self.display_slide_para(paras, sulfix)

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
            self.ROW_TAKE = int(cache.num_rows)
        else:
            self.ROW_TAKE = -1
        return self.INPUT

    def draw_graph(self, data):
        # Get num_rows and scale for getting data
        num_rows = self.ROW_TAKE
        for key, value in self.INPUT.items():
            if "Scale" in value:
                scale = self.INPUT[key]["Scale"]
            if "Label cluster" in value:
                label_cluster = self.INPUT[key]["Label cluster"]

        # Get data
        df_z = data.get_expression(num_rows=num_rows, scaler=scale)
        df_label = cache.label
        patient_id = data.patient_id

        # Cluster by labels
        if label_cluster:
            df_z, df_label, patient_id = sort_by_label(df_z, df_label, patient_id)
        patient_id = list(patient_id.squeeze())

        # Draw plot
        fig = heatmap_func(self.INPUT, df_z, df_label, patient_id)

        # Display plot
        st.plotly_chart(fig, use_container_width=True)
