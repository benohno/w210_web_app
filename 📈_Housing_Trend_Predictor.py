import streamlit as st
import pandas as pd
from time_series_model import model_for_county, add_regressor_column
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import requests


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_house = "https://assets2.lottiefiles.com/private_files/lf30_p5tali1o.json"
lottie_hello = load_lottieurl(lottie_url_house)

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
county_option = st.selectbox(
    label='Please pick a county:',
    options=('clark, nv', 'boise, id'),
    index=1)

st.markdown(county_option)

if st.button("Run model"):
    with st_lottie_spinner(lottie_hello, height=400):

        model = model_for_county(county_option, data)

        active_listing_count_regressor = add_regressor_column(
            'active_listing_count', county_option, 12, data)
        median_days_on_market_regressor = add_regressor_column(
            'median_days_on_market', county_option, 12, data)

        future = model.make_future_dataframe(periods=12, freq='MS')
        future['active_listing_count'] = active_listing_count_regressor
        future['median_days_on_market'] = median_days_on_market_regressor

        forecast = model.predict(future)

        fig = model.plot(forecast)

        st.pyplot(fig)
