"""
NASA TURBOFAN RUL PREDICTOR - MAIN PAGE
STREAMLIT APP SCAFFOLDED BY AI>>>>>>>>
"""

from PIL import Image
import streamlit as st

# Load image
#placeholder= Image.open("../../src/placeholder.jpg")
# NOT RELATIVE PATH FIX WHEN OPENING THE APP
placeholder= Image.open("src/placeholder.jpg")
st.image(placeholder, use_column_width=True)

# Set page config
st.set_page_config(
  page_title= "NASA TURBOFAN RUL PREDICTOR",
  page_icon=  placeholder,
  layout=     "wide"
)

st.markdown("""
  <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; }
    .sub-header  { font-size: 2rem; color: #4B5563; margin-bottom: 2rem; }
    .metric-card { background-color: #F3F4F6; padding: 1rem; border-radius: 10px; border-left: 5px solid #3B82F6; }
  </style>
""", unsafe_allow_html= True)

# Header
st.markdown("<h1 class= 'main-header'> NASA TURBOFAN RUL PREDICTOR </h1>", unsafe_allow_html= True)
st.markdown("<p class= 'sub-header'> Predictive Maintenance for FD001 dataset </p>", unsafe_allow_html= True)

# App states from init
if "load_data" not in st.session_state:
  st.session_state.load_data=    False
if "process_data" not in st.session_state:
  st.session_state.process_data= False
if "dashboards" not in st.session_state:
  st.session_state.dashboards=   False

# col layour for cards
col1, col2, col3= st.columns(3)

with col1:
  with st.container(border= True):
    st.markdown("### 1. Load Data")
    st.markdown("Load sensor data")
    if st.button("Go to Load Data", use_container_width=True):
      st.switch_page("pages/load_data.py")

    if st.session_state.load_data:
      st.success("Data loaded", icon="OK ✓")

with col2:
  with st.container(border= True):
    st.markdown("### 2. Process Data")
    st.markdown("Process sensor data")
    if st.button("Go to Process Data", use_container_width=True):
      st.switch_page("pages/process_data.py")

    if st.session_state.process_data:
      st.success("Data processed", icon="OK ✓")

with col3:
  with st.container(border= True):
    st.markdown("### 3. Dashboard")
    st.markdown("Results and Alerts")
    if st.button("Go to Dashboard", use_container_width= True):
      st.switch_page("pages/dashboards.py")

    if st.session_state.dashboards:
      st.switch_page("pages/dashboards.py")