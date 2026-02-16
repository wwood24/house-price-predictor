import streamlit as st
import requests
import json
import time
import os
import socket  # For hostname and IP address
import datetime as dt
from dotenv import load_dotenv

load_dotenv()

# buil the app
# Set the page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="auto"
)

# Add title and description
st.title("House Price Prediction")
st.markdown(
    """
    <p style="font-size: 18px; color: gray;">
        A MLOps project for real-time house price prediction
    </p>
    """,
    unsafe_allow_html=True,
)

# Create a two-column layout
col1, col2 = st.columns(2, gap="large")

# Input form
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Info
    st.markdown("## Submit house details to predict is price")
    # Year house built
    year_built = st.slider("**Year Built:**",
                           min_value=1800,max_value=int(dt.datetime.now().year),
                           value=2000,step=1)
    # Square Footage info
    liv_sqft_col,tot_bst_sqft_col,fin_bst_sqft_col = st.columns(3)
    st.markdown("### Provide Square foot info")
    with liv_sqft_col:
        grliv_area = st.number_input(label='Total Square feet above ground',
                                     min_value=0,value=1200,step=50)
    with tot_bst_sqft_col:
        total_bsmtsf = st.number_input(label='Total Square feet basement',
                                       min_value=0,value=1000,step=50)
    with fin_bst_sqft_col:
        bsmt_finsf = st.number_input(label='Square footage finished basement',
                                     min_value=0,value=500,step=50)
    qual_col,cond_col = st.columns(2)
    st.markdown('### Provide House Condition info')
    with qual_col:
        overall_qual = st.number_input('Overall Quality of house material',
                                       min_value=1,max_value=10,value=5)
    with cond_col:
        overall_cond = st.number_input('Overall Condition of House',
                                       min_value=1,max_value=10,value=5)
    full_bath_col,bsmt_full_bath_col,half_bath_col,bsmt_half_bath_col,bedroom_col=st.columns(5)
    st.markdown('### Provide Bedroom and Bathroom info')
    with full_bath_col:
        fullbath = st.number_input(label='Number of full baths above ground',
                                   min_value=0,value=2)
    with bsmt_full_bath_col:
        bsmtfullbath = st.number_input(label='Number of full baths in basement',
                                       min_value=0,value=1)
    with half_bath_col:
        halfbath = st.number_input(label='Number of half baths above ground',
                                   min_value=0,value=0)
    with bsmt_half_bath_col:
        bsmthalfbath=st.number_input(label='Number of half baths in basement',min_value=0,
                                     value=0)
    with bedroom_col:
        bedroomabvgr = st.number_input(label='Number of bedrooms',
                                       min_value=0,value=3)
    
    # Additional info
    house_remodel_col,garage_col,finished_garage_col=st.columns(3)
    st.markdown('### Provide Additional House Info')
    with house_remodel_col:
        house_have_remodel = st.selectbox(label='Has the house have any remodels',
                                          options=["no","yes"],index=0)
    with garage_col:
        has_garage = st.selectbox(label='Does place have a garage',
                                  options=['no','yes'],index=0)
    with finished_garage_col:
        garage_finished = st.selectbox(label='Is the garage fininished',
                                       options=['no','yes'],index=0)

    # Predict button
    predict_button = st.button("Predict Price", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Results section
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Prediction Results", unsafe_allow_html=True)

    # If button is clicked, show prediction
    if predict_button:
        # Show loading spinner
        with st.spinner("Calculating prediction..."):
            # Prepare data for API call
            api_data = {
                "YearBuilt": year_built,
                "OverallQual":overall_qual,
                "OverallCond":overall_cond,
                "house_have_remodel":house_have_remodel,
                "BsmtFinSF":bsmt_finsf,
                "TotalBsmtSF":total_bsmtsf,
                "GrLivArea":grliv_area,
                "FullBath":fullbath,
                "BsmtFullBath":bsmtfullbath,
                "HalfBath":halfbath,
                "BsmtHalfBath":bsmthalfbath,
                "BedroomAbvGr":bedroomabvgr,
                "has_garage":has_garage,
                "garage_finished":garage_finished
            }

            try:
                # Get API endpoint from environment variable or use default
                api_endpoint = os.getenv("API_URL", "http://localhost:8000")
                predict_url = f"{api_endpoint.rstrip('/')}/predict"

                st.write(f"Connecting to API at: {predict_url}")

                # Make API call to FastAPI backend
                response = requests.post(predict_url, json=api_data)
                response.raise_for_status()  # Raise exception for bad status codes
                prediction = response.json()

                # Store prediction in session state
                st.session_state.prediction = prediction
                st.session_state.prediction_time = time.time()
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")
                st.warning("Using mock data for demonstration purposes. Please check your API connection.")
                # For demo purposes, use mock data if API fails
                st.session_state.prediction = {
                    "predicted_price": 467145,
                    "confidence_interval": [420430.5, 513859.5],
                    "features_importance": {
                        "sqft": 0.43,
                        "location": 0.27,
                        "bathrooms": 0.15
                    },
                    "prediction_time": "0.12 seconds"
                }
                st.session_state.prediction_time = time.time()

    # Display prediction if available
    if "prediction" in st.session_state:
        pred = st.session_state.prediction

        # Format the predicted price
        formatted_price = "${:,.0f}".format(pred["predicted_price"])
        st.markdown(f'<div class="prediction-value">{formatted_price}</div>', unsafe_allow_html=True)
        # Display price range and prediction time
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Price Range</p>', unsafe_allow_html=True)
            lower = "${:,.1f}".format(pred["confidence_interval"][0])
            upper = "${:,.1f}".format(pred["confidence_interval"][1])
            st.markdown(f'<p class="info-value">{lower} - {upper}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Prediction Time</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="info-value">{pred["prediction_time"]} seconds</p>', 
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Top factors
        st.markdown('<div class="top-factors">', unsafe_allow_html=True)
        st.markdown("### Top Factors Affecting Price:", unsafe_allow_html=True)
        factors_dict = pred['feature_contribution']
        factors_dict = dict(sorted(factors_dict.items(),key=lambda x: x[1],reverse=True))
        final_dict = {}
        for k,v in factors_dict.items():
            if k !='expected_value':
                final_dict[k] = round(v,2)
        st.json(final_dict)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Display placeholder message
        st.markdown("""
        <div style="display: flex; height: 300px; align-items: center; justify-content: center; color: #6b7280; text-align: center;">
            Fill out the form and click "Predict Price" to see the estimated house price.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Fetch version, hostname, and IP address
version = os.getenv("APP_VERSION", "1.0.0")  # Default version if not set in environment
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

# Add footer
st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line for separation
st.markdown(
    f"""
    <div style="text-align: center; color: gray; margin-top: 20px;">
        <p><strong>Built for MLOps tranining</strong></p>
        <p>by <a href="https://github.com/wwood24/house-price-predictor" target="_blank">HousePricePredictor</a></p>
        <p><strong>Version:</strong> {version}</p>
        <p><strong>Hostname:</strong> {hostname}</p>
        <p><strong>IP Address:</strong> {ip_address}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
