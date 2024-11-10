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

# Slider inputs for nutrient range
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

# Display filtered dataset to help with debugging
st.write(f"Filtered recipes count: {len(filtered_recipes)}")
st.write(filtered_recipes[['Recipe_name', 'Cuisine_type', 'Protein(g)', 'Carbs(g)', 'Fat(g)']])

# Ensure there are enough filtered recipes before applying the model
if len(filtered_recipes) > 0:
    # Take the average of the filtered recipes (or use one recipe) to generate recommendations
    avg_recipe = filtered_recipes[['Protein(g)', 'Carbs(g)', 'Fat(g)']].mean(axis=0).values.reshape(1, -1)

    # Recommend based on the filtered average recipe
    recommended = recommend_recipe(avg_recipe)
    st.write("Recommended Recipes based on input:")
    st.dataframe(recommended[['Recipe_name', 'Cuisine_type', 'Protein(g)', 'Carbs(g)', 'Fat(g)']])
else:
    st.write("No recipes match your criteria, try adjusting your filters.")
