import streamlit as st  
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
import sklearn

# Custom CSS for Styling
background = '''
<style>
/* Background */
body {
    background-color: #f7f9fc;
    font-family: 'Arial', sans-serif;
}

/* Header */
h1, h2, h3 {
    color: #2c3e50;
    text-align: center;
}

/* Input fields */
.css-1dp5vir, .stNumberInput input {
    background-color: #ffffff !important;
    border: 1px solid #dfe6e9 !important;
    border-radius: 10px !important;
    padding: 10px !important;
    color: #2c3e50 !important;
}

/* Columns */
.stColumn {
    margin-bottom: 15px;
}

/* Button */
.stButton>button {
    background-color: #3498db;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-size: 16px;
    cursor: pointer;
}

.stButton>button:hover {
    background-color: #2980b9;
}
</style>
'''

# Apply the CSS
st.markdown(background, unsafe_allow_html=True)

# Main application UI
st.header("Online Engagement Prediction 🧑‍💻")
cola, colb, colc = st.columns(3)
with colb:
    st.image(
        "https://static.vecteezy.com/system/resources/previews/002/173/392/non_2x/student-studying-at-home-free-vector.jpg", 
        caption="Company Logo 🏫"
    )
    # Sample dataset
sample_data = {
    'CourseCategory': ['Health', 'Arts', 'Science', 'Programming', 'Business'],
    'TimeSpentOnCourse': [5, 10, 15, 20, 25],
    'NumberOfVideosWatched': [2, 4, 6, 8, 10],
    'NumberOfQuizzesTaken': [1, 2, 3, 4, 5],
    'QuizScores': [50, 60, 70, 80, 90],
    'CompletionRate': [0.5, 0.6, 0.7, 0.8, 0.9],
    'DeviceType': [0, 0, 1, 1, 0]
}
df = pd.DataFrame(sample_data)

# Display the sample dataset
st.write("### Sample Dataset Preview 📊")
st.dataframe(df.head())

# Taking X column values from the user
col1, col2 = st.columns(2)
with col1:
    # Modify CourseCategory to take unique values as input and map them
    category_mapping = {
        'Health': 0,
        'Arts': 1,
        'Science': 2,
        'Programming': 3,
        'Business': 4
    }
    unique_categories = list(category_mapping.keys())
    selected_category = st.selectbox("Select Course Category 🎓", unique_categories)
    c = category_mapping[selected_category]   

with col2:
    t = st.number_input(f"Enter Time Spent On Course ⏳ {df.TimeSpentOnCourse.min()} to Max {df.TimeSpentOnCourse.max()}")

col3, col4 = st.columns(2)
with col3:
    n = st.number_input(f"Enter Number Of Videos Watched 🎥 {df.NumberOfVideosWatched.min()} to Max {df.NumberOfVideosWatched.max()}")
    
with col4:
    q = st.number_input(f"Enter Number Of Quizzes Taken 📝 {df.NumberOfQuizzesTaken.min()} to Max {df.NumberOfQuizzesTaken.max()}")

col5, col6 = st.columns(2)
with col5:
    qs = st.number_input(f"Enter Quiz Scores 📈 {df.QuizScores.min()} to Max {df.QuizScores.max()}")
    
with col6:
    cr = st.number_input(f"Enter Completion Rate ✅ {df.CompletionRate.min()} to Max {df.CompletionRate.max()}")

# Device Type Input
d = st.number_input(f"Enter DeviceType 📱 {df.DeviceType.min()} to Max {df.DeviceType.max()}")

# Collect input data
xdata = [c, t, n, q, qs, cr, d]

# Prediction Logic

import pickle
with open('modelrf.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare the DataFrame for prediction
x = pd.DataFrame([xdata], columns=df.columns[0:7])

st.write("Given Input:")
st.dataframe(x)

# Prediction Button
if st.button("Predict 🔮"):
    prediction = model.predict(x)

    if prediction[0] == 0:
        st.success('Course Not Completed ❌') 
    elif prediction[0] == 1:
        st.success('Course Completed ✅')
    else:
        st.error('Unknown label ⚠️')