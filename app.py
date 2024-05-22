import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the logo image
logo_image = 'geni.png'

@st.cache
def load_data():
    skills_df = pd.read_excel('Skills.xlsx', engine='openpyxl')
    occupation_df = pd.read_excel('Occupation Data.xlsx', engine='openpyxl')
    return skills_df, occupation_df

@st.cache
def process_data(skills_df, occupation_df):
    # Merge the data
    merged_df = pd.merge(skills_df, occupation_df, on='O*NET-SOC Code', how='inner')
    
    # Pivot the data to get skill importance and level for each occupation
    pivot_df = merged_df.pivot_table(
        index=['O*NET-SOC Code', 'Title_y', 'Description'], 
        columns='Element Name', 
        values='Data Value', 
        aggfunc='mean'
    ).reset_index()
    
    # Extract skill-related columns
    skill_columns = pivot_df.columns[3:]
    
    # Calculate the cosine similarity between occupations
    skill_matrix = pivot_df[skill_columns].fillna(0)
    cosine_sim = cosine_similarity(skill_matrix)
    
    # Convert the similarity matrix to a DataFrame
    similarity_df = pd.DataFrame(cosine_sim, index=pivot_df['Title_y'], columns=pivot_df['Title_y'])
    
    return pivot_df, similarity_df

def recommend_career_paths(occupation, similarity_df, top_n=5):
    if occupation in similarity_df.index:
        similar_occupations = similarity_df[occupation].sort_values(ascending=False).head(top_n + 1)
        similar_occupations = similar_occupations.drop(occupation)
        return similar_occupations
    else:
        return None

# Load and process data
skills_df, occupation_df = load_data()
pivot_df, similarity_df = process_data(skills_df, occupation_df)

# Streamlit app
st.image(logo_image, width=200)
st.title("Career Path Recommendation System")

st.markdown("""
### About the Dataset
The dataset used in this app consists of skills and occupation data sourced from reliable databases, including O*NET. Each occupation is associated with various skills, with importance and level values provided for each skill. 

### References
1. O*NET Database: [O*NET Online](https://www.onetonline.org/)
2. U.S. Department of Labor: [U.S. Department of Labor](https://www.dol.gov/)

### Model Used
This app uses the Cosine Similarity model to calculate the similarity between different occupations based on their skill requirements. This allows us to recommend career paths that require similar skill sets.

### Disclaimer
This application is a demo and should be used for informational purposes only. The recommendations provided are based on the available data and are meant to serve as a guide. Users are advised to perform further research and consider additional factors when making career decisions.

### Team Acknowledgement
This project is the output of the GENI research team. Our team is dedicated to providing insights and tools to help individuals navigate their career paths.

### Instructions
1. Select your current occupation from the dropdown menu.
2. Click the "Get Career Path Recommendations" button to view recommended career paths and skill gaps.
""")

# User input for current occupation
occupation = st.selectbox("Select your current occupation:", similarity_df.index)

if st.button("Get Career Path Recommendations"):
    # Get recommended career paths
    recommended_paths = recommend_career_paths(occupation, similarity_df, top_n=5)
    
    if recommended_paths is not None:
        st.write(f"Recommended career paths for {occupation}:")
        for idx, (rec_occ, score) in enumerate(recommended_paths.items(), start=1):
            st.write(f"{idx}. {rec_occ} (Similarity Score: {score:.3f})")

        # Highlight skill gaps for the top recommended career path
        top_recommended_occupation = recommended_paths.index[0]
        current_occupation_skills = pivot_df[pivot_df['Title_y'] == occupation].iloc[:, 3:].reset_index(drop=True)
        recommended_occupation_skills = pivot_df[pivot_df['Title_y'] == top_recommended_occupation].iloc[:, 3:].reset_index(drop=True)
        skill_gaps = recommended_occupation_skills - current_occupation_skills
        skill_gaps = skill_gaps.T  # Transpose for better readability
        skill_gaps.columns = ['Skill Gap']
        skill_gaps_sorted = skill_gaps.sort_values(by='Skill Gap', ascending=False)

        st.write(f"\nSkill gaps to transition from {occupation} to {top_recommended_occupation}:")
        st.table(skill_gaps_sorted.head(10))
    else:
        st.write("Occupation not found in the dataset.")
