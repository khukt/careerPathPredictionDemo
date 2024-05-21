import streamlit as st
import numpy as np
import joblib

# Explanation of the scale
scale_explanation = """
Please use the following scale to respond to each statement:
1 - Strongly Disagree
2 - Disagree
3 - Neutral
4 - Agree
5 - Strongly Agree
"""

# Title of the app
st.title('Career Prediction Assessment')

# Display the scale explanation
st.markdown(scale_explanation)

# Function to create questions with Likert scale
def create_questions(questions):
    responses = []
    for question in questions:
        response = st.radio(question, ('1', '2', '3', '4', '5'))
        responses.append(int(response))
    return responses

# Define questions for each personality trait
openness_qs = [
    "I enjoy trying new and different activities.",
    "I am imaginative and creative.",
    "I like to explore new ideas and concepts.",
    "I am open to experiencing new things."
]
conscientiousness_qs = [
    "I am always prepared and organized.",
    "I pay attention to details.",
    "I follow a schedule and complete tasks on time.",
    "I am reliable and dependable."
]
extraversion_qs = [
    "I feel comfortable around people and enjoy social gatherings.",
    "I am talkative and outgoing.",
    "I am energetic and enthusiastic.",
    "I seek out and enjoy being the center of attention."
]
agreeableness_qs = [
    "I am compassionate and have a soft heart.",
    "I am trusting and cooperative.",
    "I am helpful and willing to assist others.",
    "I am considerate and kind to almost everyone."
]
neuroticism_qs = [
    "I often feel anxious and stressed.",
    "I get upset easily and am prone to mood swings.",
    "I worry about many different things.",
    "I am easily disturbed and troubled."
]

# Personality Traits
st.header("Personality Traits")

O_score = np.mean(create_questions(openness_qs))
C_score = np.mean(create_questions(conscientiousness_qs))
E_score = np.mean(create_questions(extraversion_qs))
A_score = np.mean(create_questions(agreeableness_qs))
N_score = np.mean(create_questions(neuroticism_qs))

# Aptitude Tests
st.header("Aptitude Tests")

# Numerical Aptitude
st.subheader("Numerical Aptitude")
numerical_qs = [
    "What is 25% of 200?",
    "Solve the equation: 3x + 5 = 20.",
    "What is the next number in the series: 2, 5, 8, 11, ...?"
]
numerical_responses = []
for question in numerical_qs:
    response = st.text_input(question)
    numerical_responses.append(response)

# Spatial Aptitude
st.subheader("Spatial Aptitude")
spatial_qs = [
    "Identify the shape that can be formed by folding a given net (2D to 3D visualization).",
    "Choose the correct rotated version of a given shape.",
    "Complete a pattern by choosing the correct shape that fits into the sequence."
]
spatial_responses = []
for question in spatial_qs:
    response = st.selectbox(question, ['Option A', 'Option B', 'Option C', 'Option D'])
    spatial_responses.append(response)

# Perceptual Aptitude
st.subheader("Perceptual Aptitude")
perceptual_qs = [
    "Find the differences between two similar images.",
    "Identify the missing piece in a puzzle based on a pattern.",
    "Recognize a hidden shape within a complex design."
]
perceptual_responses = []
for question in perceptual_qs:
    response = st.selectbox(question, ['Option A', 'Option B', 'Option C', 'Option D'])
    perceptual_responses.append(response)

# Abstract Reasoning
st.subheader("Abstract Reasoning")
abstract_qs = [
    "Determine the next figure in a sequence of shapes based on a pattern.",
    "Solve analogies: A is to B as C is to __.",
    "Identify the odd one out in a series of figures based on a rule."
]
abstract_responses = []
for question in abstract_qs:
    response = st.selectbox(question, ['Option A', 'Option B', 'Option C', 'Option D'])
    abstract_responses.append(response)

# Verbal Reasoning
st.subheader("Verbal Reasoning")
verbal_qs = [
    "Read a passage and answer questions about its content.",
    "Determine the meaning of a word based on context.",
    "Choose the correct conclusion based on a given set of statements."
]
verbal_responses = []
for question in verbal_qs:
    response = st.selectbox(question, ['Option A', 'Option B', 'Option C', 'Option D'])
    verbal_responses.append(response)

# Function to calculate aptitude scores
def calculate_aptitude_scores(numerical_responses, spatial_responses, perceptual_responses, abstract_responses, verbal_responses):
    Numerical_Aptitude = sum([int(num) if num.isdigit() else 0 for num in numerical_responses])
    Spatial_Aptitude = sum([1 for spatial in spatial_responses if spatial == 'Option A'])  # Example scoring logic
    Perceptual_Aptitude = sum([1 for perceptual in perceptual_responses if perceptual == 'Option A'])  # Example scoring logic
    Abstract_Reasoning = sum([1 for abstract in abstract_responses if abstract == 'Option A'])  # Example scoring logic
    Verbal_Reasoning = sum([1 for verbal in verbal_responses if verbal == 'Option A'])  # Example scoring logic
    return Numerical_Aptitude, Spatial_Aptitude, Perceptual_Aptitude, Abstract_Reasoning, Verbal_Reasoning

# Calculate aptitude scores on user input
if st.button('Calculate Aptitude Scores'):
    scores = calculate_aptitude_scores(numerical_responses, spatial_responses, perceptual_responses, abstract_responses, verbal_responses)
    st.write("Aptitude Scores: Numerical Aptitude={}, Spatial Aptitude={}, Perceptual Aptitude={}, Abstract Reasoning={}, Verbal Reasoning={}".format(*scores))
    st.session_state['scores'] = scores

# Combine all scores and use them for prediction
if st.button('Predict Career'):
    if 'scores' in st.session_state:
        scores = st.session_state['scores']
        personality_scores = np.array([O_score, C_score, E_score, A_score, N_score])
        aptitude_scores = np.array(scores)
        combined_scores = np.concatenate((personality_scores, aptitude_scores)).reshape(1, -1)

        # Load the trained model and scaler
        clf = joblib.load('career_prediction_model.pkl')  # Ensure the model file is in the same directory
        scaler = joblib.load('scaler.pkl')  # Ensure the scaler file is in the same directory

        # Scale the combined scores
        combined_scores_scaled = scaler.transform(combined_scores)

        # Make prediction
        predicted_career = clf.predict(combined_scores_scaled)
        st.write(f"Predicted Career: {predicted_career[0]}")
    else:
        st.write("Please calculate aptitude scores first.")
