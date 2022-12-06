import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import pandas as pd


st.set_page_config(
    page_title="Housing Trend Predictor",
    page_icon="🏘️",
    initial_sidebar_state="collapsed"
)

st.image("img/housing-trend-predictor-black-white.png")

# title and introductory text
st.title('Housing Trend Predictor')

with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """
-   The *Housing Trend Predictor* app is an easy-to-use interface built in Streamlit to forecast direction of the housing market for a chosen US county
-   It uses a Prophet timeseries model to make predictions from housing and market data the team has collected
	    """
    )

    st.markdown("")

# read in data

data = pd.read_csv('data/cleaned_data.csv')

counties_only = pd.DataFrame(data.county_name.unique(), columns=['Counties'])

# user inputs
option = st.selectbox(
    'Please pick a county:',
    ('clark, nv', 'boise, id'))

st.markdown(option)
