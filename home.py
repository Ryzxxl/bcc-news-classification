# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url("logo\logo.png");
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 25px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "HOME";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()


html_temp = """
<div style ="background-color:#758283;padding:13px">
<h1 style ="color:black;text-align:center;"> HOME !🏠 </h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)

st.image('logo/logo.png', width=705)



st.sidebar.subheader("By")
st.sidebar.text("Rakshit Khajuria - 19bec109")
st.sidebar.text("Prikshit Sharma - 19bec062")
