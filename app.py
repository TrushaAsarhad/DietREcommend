import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model
model = joblib.load('diet_recommender_model.pkl')

# Load the dataset (ensure this is available in your app directory)
df = pd.read_csv('All_Diets.csv')

# Function to recommend recipes based on user input
def recommend_recipe(user_recipe):
    X = df[['Protein(g)', 'Carbs(g)', 'Fat(g)']]
    distances, indices = model.kneighbors(user_recipe)
    return df.iloc[indices[0]]

# Streamlit app interface
st.title("Diet Recommender System")

# User input section
diet_type_input = st.selectbox('Select Diet Type:', df['Diet_type'].unique())
cuisine_input = st.selectbox('Select Cuisine Type:', df['Cuisine_type'].unique())

protein_min = st.slider('Min Protein (g):', 0, 50, 10)
protein_max = st.slider('Max Protein (g):', 0, 50, 30)
carbs_min = st.slider('Min Carbs (g):', 0, 100, 20)
carbs_max = st.slider('Max Carbs (g):', 0, 100, 50)
fat_min = st.slider('Min Fat (g):', 0, 50, 5)
fat_max = st.slider('Max Fat (g):', 0, 50, 20)

# Filter dataset based on user input
filtered_recipes = df[
    (df['Diet_type'] == diet_type_input) & 
    (df['Cuisine_type'] == cuisine_input) & 
    (df['Protein(g)'] >= protein_min) & (df['Protein(g)'] <= protein_max) &
    (df['Carbs(g)'] >= carbs_min) & (df['Carbs(g)'] <= carbs_max) &
    (df['Fat(g)'] >= fat_min) & (df['Fat(g)'] <= fat_max)
]

# Display recommendations
if st.button("Get Recommendations"):
    user_recipe = np.array([[protein_min, carbs_min, fat_min]])
    recommended = recommend_recipe(user_recipe)
    st.write("Recommended Recipes:")
    st.dataframe(recommended[['Recipe_name', 'Cuisine_type', 'Protein(g)', 'Carbs(g)', 'Fat(g)']])

