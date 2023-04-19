from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.options.mode.chained_assignment = None


def match_column_with_expression_index(df_exp, df_clin):
    # define the list of values to search for
    search_list = list(df_exp.index)

    # create a boolean mask for each column indicating which rows contain any of the given values
    mask = df_clin.apply(lambda x: x.isin(search_list))

    # sum the number of True values in each column to get the total count of times the given values appear in that column
    counts = mask.sum()

    # find the column with the highest count
    matched_col = counts.idxmax()

    # print the result
    print("\nThe column picked to merge clinical and expression data: ", matched_col)
    return matched_col


def handle_unknown_values(df, unique_values, col):
    for value in unique_values:
        try:
            if "unknown" in value or "Na" in value or "NaN" in value or "None" in value:
                df[col][df[col] == value] = np.NaN
                unique_values.remove(value)
        except Exception as e:
            pass
        try:
            unique_values.remove(np.NaN)
        except:
            pass
        try:
            unique_values.remove(None)
        except:
            pass
    return unique_values


def handle_binary_column(df, unique_values, column, denote_log):
    try:
        first_value, sec_value = unique_values
        if "no" in first_value or "negative" in first_value or "without" in first_value:
            df[column][df[column] == first_value] = 0
            df[column][df[column] == sec_value] = 1
            denote_log[column] = {
                0: first_value,
                1: sec_value,
            }
        else:
            df[column][df[column] == first_value] = 1
            df[column][df[column] == sec_value] = 0
            denote_log[column] = {
                0: sec_value,
                1: first_value,
            }

        df[column] = pd.to_numeric(df[column])
    except:
        pass
    return denote_log


def is_numeric_column(df, column):
    try:
        df[column] = pd.to_numeric(df[column])
        return True
    except ValueError:
        return False


def is_numeric_in_unclassified_column(df, column, threshold):
    non_num = []
    for i in df[column]:
        try:
            pd.to_numeric(i)
        except:
            non_num.append(i)
    non_num_ratio = len(non_num) / len(df[column])

    if non_num_ratio <= threshold:
        unique_non_num = set(non_num)
        print(f"\nOmited {unique_non_num} in {column}")
        for value in unique_non_num:
            df[column][df[column] == value] = np.NaN
        df[column] = pd.to_numeric(df[column])
        return True
    else:
        return False


def rename_numeric_binary_cols(df):
    binary_lst = []
    numeric_lst = []
    denote_log = {}
    for col in df.columns:
        # Define unique_values of a column
        unique_values = list(set(df[col]))

        # Format unknown value
        unique_values = handle_unknown_values(df, unique_values, col)

        # Format binary value (1, 0) and rename binary column
        if len(unique_values) == 2:
            binary_lst.append(col)
            denote_log = handle_binary_column(df, unique_values, col, denote_log)
            df.rename(columns={col: f"binary_{col}"}, inplace=True)

        # Rename numeric column
        elif is_numeric_column(df, col):
            numeric_lst.append(col)
            df.rename(columns={col: f"numeric_{col}"}, inplace=True)

        # Rename unclassified column
        elif is_numeric_in_unclassified_column(df, col, threshold=0.1):
            numeric_lst.append(col)
            df.rename(columns={col: f"numeric_{col}"}, inplace=True)
        else:
            df.rename(columns={col: f"info_{col}"}, inplace=True)
            # df = df.drop(columns=col)
    df = df.sort_index(axis=1)
    return df, denote_log


@dataclass
class PlotData:
    merged_data = None

    clinical_data: DataFrame
    patient_id = None
    binary_data = None
    numeric_data = None
    info_data = None
    labels = None

    expression_data: DataFrame
    gene_lst = None

    denote_log = None

    def clean_clinical(self):
        self.clinical_data, denote_log = rename_numeric_binary_cols(self.clinical_data)
        self.denote_log = pd.DataFrame(denote_log).T

    def clean_expression(self):
        self.gene_lst = self.expression_data.iloc[:, 0]
        self.expression_data = self.expression_data.transpose()
        self.expression_data = self.expression_data.rename(
            columns=self.expression_data.iloc[0, :]
        ).iloc[1:, :]
        self.expression_data = self.expression_data.apply(
            pd.to_numeric, errors="coerce"
        )

    # Find column that match the index of expression data
    def pick_similar_column(self):
        matched_col = match_column_with_expression_index(
            self.expression_data, self.clinical_data
        )
        self.clinical_data = self.clinical_data.set_index(matched_col)
        self.clinical_data.index.name = None

    def drop_and_return_column(self, prefix):
        col_lst = [col for col in self.merged_data.columns if col.startswith(prefix)]
        cols = self.merged_data[col_lst]
        self.merged_data = self.merged_data.drop(columns=col_lst)
        return cols.T

    def generate_data(self):
        # Preprocess expression data
        self.clean_expression()

        # Clean clinical data
        self.clean_clinical()

        # Pick similar column for merging
        self.pick_similar_column()

        # Merge clinical and expression data
        self.merged_data = self.clinical_data.merge(
            self.expression_data, right_index=True, left_index=True
        )
        self.merged_data = self.merged_data.reset_index()

        # Take out expression data
        self.expression_data = self.merged_data[self.gene_lst].T
        self.merged_data = self.merged_data.drop(columns=self.gene_lst)

        # Take out different clinical data
        self.patient_id = self.drop_and_return_column("index")
        self.binary_data = self.drop_and_return_column("binary")
        self.numeric_data = self.drop_and_return_column("numeric")
        self.info_data = self.drop_and_return_column("info")
        self.labels = pd.concat(
            [self.binary_data, self.numeric_data, self.info_data], axis=0
        )

    def get_expression(self, num_rows: int = -1, scaler="Standard"):
        df = self.expression_data
        if scaler == "Standard":
            scaler = StandardScaler()
        elif scaler == "Min Max":
            scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df.T), columns=df.index).T
        # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        if num_rows == -1:
            return df
        else:
            return df.iloc[:num_rows, :]

    def get_label(self, label_name: str = None):
        # label = [
        #     rownames for rownames in self.labels.index if rownames.endswith(label_name)
        # ]
        df_label = self.labels.loc[[label_name]]
        new_name = label_name.split("_", maxsplit=1)[1]
        df_label.rename(index={label_name: new_name}, inplace=True)
        return df_label
