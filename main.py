import streamlit as st
import pandas as pd
from web_scraper import scrape_privacy_policy_url
from policy_scores import readability_score, word_count

st.set_page_config(
    page_title="Privacy Policy Analyzer",
    page_icon="🎈",
)

st.image("img/privacy-dash-logo.png")

# title and introductory text
st.title('Privacy Policy Analyzer')

with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   The *Privacy Policy Analyzer* app is an easy-to-use interface built in Streamlit to help users understand high level risks of a privacy policy
-   It uses a custom parser that leverages NLP and other language packages
	    """
    )

    st.markdown("")

# user inputs
st.markdown("## **📌 Paste Privacy Policy URL **")

privacy_policy_url = st.text_input("Here:")

# get text from url
privacy_policy_str = scrape_privacy_policy_url(privacy_policy_url)

st.write('The url input is: ', privacy_policy_url)

# initial values before user inputs url
if privacy_policy_url == '':
    reading_score = 'NA'
    word_total = 'NA'
    time_to_read = 'NA'

else:
    reading_score = readability_score(privacy_policy_str)
    word_total = word_count(privacy_policy_str)
    time_to_read = round(word_total/250)

# metrics display
col1, col2, col3 = st.columns(3)
col1.metric("Reading Grade Level", reading_score)
col2.metric("Total Length", str(word_total) + ' words')
col3.metric("Estimated Time to Read", str(time_to_read) + ' minutes')

# flags display
st.text("🚩 - Risk #1")
st.text("🚩 - Risk #2")
st.text("🚩 - Risk #3")
