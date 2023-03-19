import pandas as pd
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()
RESULT_PATH=os.getenv("result_path")
geo_id = "GSE157103"

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
    print("The column with the most occurrences of the given values is: ", matched_col)
    return matched_col

def unknown_value(df, unique_values, col):
    for value in unique_values:
        try:
            if ("unknown" in value
                or "Na" in value
                or "NaN" in value):
                df[col][df[col] == value]= None
                unique_values.remove(value)
        except Exception as e:
            pass
    return unique_values


# Load data from .csv file in processed_data
clinical_data = pd.read_csv(f"{RESULT_PATH}/{geo_id}/{geo_id}_clinical.csv")
expression_data = pd.read_csv(f"{RESULT_PATH}/{geo_id}/{geo_id}_expression.csv")
expression_data = expression_data.transpose()
expression_data = (
    expression_data.rename(columns=expression_data.iloc[0, :])
    .iloc[1:, :]
)


expression_data=expression_data.iloc[:,:10]
# st.write(type(expression_data.index))
# Find column that match the index of expression data
matched_col = match_column_with_expression_index(expression_data, clinical_data)

clinical_data = clinical_data.set_index(matched_col)
clinical_data.index.name = None




# print(list(set(clinical_data["age (years)"])))


pd.options.mode.chained_assignment = None  # default='warn'
for col in clinical_data.columns:
    unique_values = list(set(clinical_data[col]))
    unique_values = unknown_value(clinical_data, unique_values, col)
    if len(unique_values) == 2:
        first_value = unique_values[0]
        if ("no" in first_value
            or "negative" in first_value
            or "without" in first_value):
            clinical_data[col][clinical_data[col] == first_value]        = 0
            clinical_data[col][clinical_data[col] == unique_values[1]]   = 1
        else:
            clinical_data[col][clinical_data[col] == first_value]        = 1
            clinical_data[col][clinical_data[col] == unique_values[1]]   = 0

merged_data=clinical_data.merge(expression_data, right_index=True, left_index= True)

st.write(clinical_data)
st.write(expression_data)
st.write(merged_data)






# # - Line chart:
# import streamlit as st
# import pandas as pd
# import altair as alt
  
# data = pd.read_csv("<filename.csv>")
  
# chart = alt.Chart(data).mark_line().encode(
#     x='x_variable',
#     y='y_variable').interactive()
    
# st.write(chart)

# # - Scatter plot:
# import streamlit as st
# import pandas as pd
# import altair as alt
  
# data = pd.read_csv("<filename.csv>")
  
# chart = alt.Chart(data).mark_circle().encode(
#     x='x_variable',
#     y='y_variable',
#     color='category').interactive()
    
# st.write(chart)

# # - Bar chart:
# import streamlit as st
# import pandas as pd
# import altair as alt
  
# data = pd.read_csv("<filename.csv>")
  
# chart = alt.Chart(data).mark_bar().encode(
#     x='x_variable',
#     y='count()',
#     color='category').interactive()
    
# st.write(chart)

# # - Histogram:
# import streamlit as st
# import pandas as pd
# import altair as alt
  
# data = pd.read_csv("<filename.csv>")
  
# chart = alt.Chart(data).mark_bar().encode(
#     x='bin_start',
#     y='count()').interactive()
    
# st.write(chart)

# # - Heatmap:
# import streamlit as st
# import pandas as pd
# import seaborn as sns
  
# data = pd.read_csv("<filename.csv>")
  
# heatmap = sns.heatmap(data.corr())
    
# st.pyplot(heatmap)
