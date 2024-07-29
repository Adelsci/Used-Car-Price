import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load your trained model
model = joblib.load('models/car_price_model.pkl')

# Load encoders and scaler
encoders = joblib.load('models/encoders.pkl')
scaler = joblib.load('models/scaler.pkl')

# Function to encode inputs
def encode_input(data, encoders):
    for column, encoder in encoders.items():
        data[column] = encoder.transform([data[column]])[0]
    return data

# Load the original dataset for visualizations
Car_listings = pd.read_csv('C:\deploy streamlit/true_car_listings.csv')
Car_listings = Car_listings.drop(columns=['Vin'])

# Calculate IQR for Mileage and Price
Q1_mileage = Car_listings['Mileage'].quantile(0.25)
Q3_mileage = Car_listings['Mileage'].quantile(0.75)
IQR_mileage = Q3_mileage - Q1_mileage
lower_bound_mileage = Q1_mileage - 1.5 * IQR_mileage
upper_bound_mileage = Q3_mileage + 1.5 * IQR_mileage

Q1_price = Car_listings['Price'].quantile(0.25)
Q3_price = Car_listings['Price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price

# Filter out the outliers
data = Car_listings[(Car_listings['Mileage'] >= lower_bound_mileage) & 
                    (Car_listings['Mileage'] <= upper_bound_mileage) & 
                    (Car_listings['Price'] >= lower_bound_price) & 
                    (Car_listings['Price'] <= upper_bound_price)]

# Streamlit app
st.title('Car Price Prediction Dashboard')

# Input fields for user
marke = st.selectbox('Marke', encoders['Marke'].classes_)
model_car = st.selectbox('Model', encoders['Model'].classes_)
year = st.slider('Year', 1997, 2018, 2014)
mileage = st.number_input('Mileage', min_value=0, max_value=300000, value=50000)
city = st.selectbox('City', encoders['City'].classes_)
state = st.selectbox('State', encoders['State'].classes_)

# Create a dataframe from the inputs
input_data = {'Marke': marke, 'Model': model_car, 'Year': year, 'Mileage': mileage, 'City': city, 'State': state}
input_df = pd.DataFrame([input_data])

# Encode and scale input data
input_df = encode_input(input_df, encoders)
input_df[['Mileage']] = scaler.transform(input_df[['Mileage']])

# Predict price
predicted_price = model.predict(input_df)[0]

st.write(f'Predicted Price: ${predicted_price:.2f}')

# Additional features
st.subheader('Additional Features')

# 1. Display Histogram of Car Prices
st.subheader('Histogram of Car Prices')
fig = px.histogram(data, x='Price', nbins=50, title='Distribution of Car Prices')
st.plotly_chart(fig)

# 2. Scatter Plot of Mileage vs. Price
st.subheader('Scatter Plot of Mileage vs. Price')
fig = px.scatter(data, x='Mileage', y='Price', title='Mileage vs. Price')
st.plotly_chart(fig)

# 3. Interactive Map of Car Listings
# Load the geographical data
geo_data = pd.read_csv('C:\deploy streamlit/uscities.csv')

# Standardize the City and State columns to lowercase and remove any trailing spaces
data['City'] = data['City'].str.lower().str.strip()
data['State'] = data['State'].str.lower().str.strip()
geo_data['City'] = geo_data['City'].str.lower().str.strip()
geo_data['State'] = geo_data['State'].str.lower().str.strip()

# Create City_State column
data['City_State'] = data['City'] + ', ' + data['State']
geo_data['City_State'] = geo_data['City'] + ', ' + geo_data['State']

# Merge the car listings data with the geographical data
merged_data = pd.merge(data, geo_data[['City_State', 'lat', 'lon']], on='City_State', how='left')

# Drop rows with missing values
cleaned_data = merged_data.dropna(subset=['lat', 'lon'])

st.subheader('Map of Car Listings')
# Assuming we have latitude and longitude data for cities
fig = px.scatter_mapbox(cleaned_data, lat='lat', lon='lon', size='Price', hover_name='City_State', zoom=3, height=300)
fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig)

# Optional: Display input data
st.subheader('Input Data')
st.write(input_df)