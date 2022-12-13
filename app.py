from operator import index
import streamlit as st
import numpy as np
#from scipy._lib._util import _valarray
import plotly.express as px
from pycaret.classification import setup, compare_models, pull, save_model, load_model
# from pycaret.regression import setup, compare_models, pull, save_model, load_model
# from pycaret.clustering import setup, compare_models, pull, save_model, load_model
# from pycaret.anomaly import setup, compare_models, pull, save_model, load_model
# from pycaret.nlp import setup, compare_models, pull, save_model, load_model

#Set streamlit config theme to light color
st.set_page_config(layout="wide", page_title="AutoBrillianceML", page_icon="https://scontent-hou1-1.xx.fbcdn.net/v/t39.30808-6/301179207_173952721862947_5470447139486339930_n.png?_nc_cat=106&ccb=1-7&_nc_sid=09cbfe&_nc_ohc=ro-KbelKHRIAX8gkdAq&_nc_ht=scontent-hou1-1.xx&oh=00_AfBNS6HNHIADEANhmMB9FPCMIRueDxu2wEOx3jZB_vjPYg&oe=639C5EDE")
st.markdown("""
<style>
body, css-ffhzg2, css-1vencpc {
    background-color: #FFFFFFF !important;
    color: #000000 !important;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 20px;
    font-weight: 400;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;

}

</style>
""", unsafe_allow_html=True)


import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 


if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://scontent-hou1-1.xx.fbcdn.net/v/t39.30808-6/301179207_173952721862947_5470447139486339930_n.png?_nc_cat=106&ccb=1-7&_nc_sid=09cbfe&_nc_ohc=ro-KbelKHRIAX8gkdAq&_nc_ht=scontent-hou1-1.xx&oh=00_AfBNS6HNHIADEANhmMB9FPCMIRueDxu2wEOx3jZB_vjPYg&oe=639C5EDE")
    st.title("AutoBrillianceML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True, use_gpu=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")