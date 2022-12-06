import streamlit as st

st.set_page_config(layout="wide")

st.title('Introducing the Team')

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.header("Blake Bormes")
    st.image("img/blake_linkedin.jpeg")
    st.header("Senior Consultant")

with col2:
    st.header("Jonathan Moges")
    st.image("img/jonathan_linkedin.jpeg")
    st.header("Data Scientist")

with col3:
    st.header("Benjamin Mok")
    st.image("img/mok_picture.png")
    st.header("Mr. Due Tomorrow = Do Tomorrow")
    st.header("jk luv you bro :)")
    st.header('ML Engineer')

with col4:
    st.header("Benjamin Ohno")
    st.image("img/ohno_linkedin.jpeg")
    st.header("Data Scientist")
