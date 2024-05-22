import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
st.title("Career Path Recommendation System")

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
