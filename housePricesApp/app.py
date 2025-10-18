import streamlit as st # Import Streamlit for building the web app interface
import pandas as pd
import joblib # Import joblib for loading the saved ML model and scaler

#  Load the trained machine learning model and the feature scaler
model_data = joblib.load('house_prices_model.pkl') # Load serialized file (contains both model and scaler)
model = model_data['model'] # Extract the trained regression model
scaler = model_data['scaler']  # Extract the scaler used during training

# App title and introduction text
st.title("🏠 House Price Prediction App")
st.markdown("""
This app predicts **house prices** based on your input features.
Enter the details below and click **Predict Price** to see the result.
""")

# Section for Input fields
st.subheader("🏡 Input House Details")

# Create 3 responsive columns to organize inputs side-by-side
col1, col2, col3 = st.columns(3)
# Numeric inputs for area and room details
with col1:
    square_feet = st.number_input("Square Feet", min_value=50.0, max_value=500.0, value=150.0)
    num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
# Additional house features
with col2:
    num_floors = st.number_input("Number of Floors", min_value=1, max_value=5, value=2)
    year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
    garage_size = st.number_input("Garage Size", min_value=0, max_value=100, value=20)
# Categorical and continuous inputs
with col3:
    has_garden = st.selectbox("Has Garden", ["No", "Yes"]) # Dropdown for Yes/No
    has_pool = st.selectbox("Has Pool", ["No", "Yes"])     # Dropdown for Yes/No
    location_score = st.slider("Location Score", 1.0, 10.0, 5.0) # Slider for quality of location
    distance_to_center = st.number_input("Distance to City Center (km)", min_value=0.0, max_value=50.0, value=10.0)

# Convert categorical inputs into numeric form for model
has_garden = 1 if has_garden == "Yes" else 0
has_pool = 1 if has_pool == "Yes" else 0

# Prepare features -- Combine all inputs into a single DataFrame for prediction
features = pd.DataFrame({
    'Square_Feet': [square_feet],
    'Num_Bedrooms': [num_bedrooms],
    'Num_Bathrooms': [num_bathrooms],
    'Num_Floors': [num_floors],
    'Year_Built': [year_built],
    'Has_Garden': [has_garden],
    'Has_Pool': [has_pool],
    'Garage_Size': [garage_size],
    'Location_Score': [location_score],
    'Distance_to_Center': [distance_to_center]
})

#  Apply the same scaling transformation as used during training
scaled_features = scaler.transform(features)

# Prediction -- When user clicks "Predict Price", model generates price output
if st.button("Predict Price"):
    prediction = model.predict(scaled_features)
    st.success(f"💰 Estimated House Price: {prediction[0]:,.4f}")

# Embed Tableau dashboard

# Tableau Public dashboard URL
st.title("House Prices Tableau Dashboard")
st.markdown("""
Use the **"Year"** slider on the dashboard to interactively update the visualizations based on the selected year.   
""")

# Replace the URL below with your own
tableau_url = "https://public.tableau.com/views/realEstate_17592102423330/HousePriceDashboard?:showVizHome=no&:embed=true"

# Display the Tableau dashboard inside the Streamlit app
st.components.v1.iframe(tableau_url, width=1000, height=827)
