import streamlit as st
import pandas as pd
# from time_series_model import model_for_county, add_regressor_column
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import requests
from prophet import Prophet


def model_for_county(county, data):
    """

    """
    county_data = data.loc[data.county_name == county][['date', 'median_listing_price',
                                                        'active_listing_count', 'median_days_on_market']]

    # adding lockdowns as holidays to prevent model from trying to model anomoly data
    lockdowns = pd.DataFrame([
        {'holiday': 'lockdown_1', 'ds': '2020-03-21',
            'lower_window': 0, 'ds_upper': '2020-06-06'},
        {'holiday': 'lockdown_2', 'ds': '2020-07-09',
            'lower_window': 0, 'ds_upper': '2020-10-27'},
        {'holiday': 'lockdown_3', 'ds': '2021-02-13',
            'lower_window': 0, 'ds_upper': '2021-02-17'},
        {'holiday': 'lockdown_4', 'ds': '2021-05-28',
            'lower_window': 0, 'ds_upper': '2021-06-10'},
    ])

    for t_col in ['ds', 'ds_upper']:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns['upper_window'] = (
        lockdowns['ds_upper'] - lockdowns['ds']).dt.days

    county_data.columns = ['ds', 'y', 'active_listing_count',
                           'median_days_on_market']

    # instantiate Prophet model object with confidence interval with of 95%
    m = Prophet(seasonality_mode='multiplicative',
                holidays=lockdowns, interval_width=0.95)

    m = Prophet(seasonality_mode='multiplicative', interval_width=0.95)

    # adding additional regressor
    m.add_regressor('active_listing_count')
    m.add_regressor('median_days_on_market')

    m.fit(county_data)

    return m


def add_regressor_column(regressor_name, county, months_to_forecast, data):
    county_data = data.loc[data.county_name ==
                           county][['date', regressor_name]]

    # adding lockdowns as holidays to prevent model from trying to model anomoly data
    lockdowns = pd.DataFrame([
        {'holiday': 'lockdown_1', 'ds': '2020-03-21',
            'lower_window': 0, 'ds_upper': '2020-06-06'},
        {'holiday': 'lockdown_2', 'ds': '2020-07-09',
            'lower_window': 0, 'ds_upper': '2020-10-27'},
        {'holiday': 'lockdown_3', 'ds': '2021-02-13',
            'lower_window': 0, 'ds_upper': '2021-02-17'},
        {'holiday': 'lockdown_4', 'ds': '2021-05-28',
            'lower_window': 0, 'ds_upper': '2021-06-10'},
    ])

    for t_col in ['ds', 'ds_upper']:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns['upper_window'] = (
        lockdowns['ds_upper'] - lockdowns['ds']).dt.days

    county_data.columns = ["ds", "y"]

    # instantiate Prophet model object with confidence interval with of 95%
    m = Prophet(seasonality_mode='multiplicative',
                holidays=lockdowns, interval_width=0.95)

    m.fit(county_data)

    future = m.make_future_dataframe(periods=12, freq='MS')

    forecast = m.predict(future)

    return forecast.yhat


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
