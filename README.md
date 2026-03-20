# 🎓 Online Course Engagement Prediction

A machine learning project that predicts whether a student will complete an online course based on their engagement behaviour. The project includes a full ML pipeline in a Jupyter notebook and an interactive web application built with Streamlit.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Model Comparison](#model-comparison)
- [Web Application](#web-application)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)

---

## Project Overview

With the growing popularity of online learning, understanding student engagement patterns is crucial for educators and platform providers. This project aims to **predict course completion** (binary classification: `0` = Not Completed, `1` = Completed) using features that describe a student's activity on an online learning platform.

The end product is a Streamlit web app where users can enter engagement metrics and instantly get a prediction.

---

## Dataset

**File:** `online_course_engagement_data.csv`  
**Size:** ~9,000 rows

| Column | Description |
|---|---|
| `UserID` | Unique identifier for each student (dropped before modelling) |
| `CourseCategory` | Category of course: Health, Arts, Science, Programming, Business |
| `TimeSpentOnCourse` | Total hours spent on the course |
| `NumberOfVideosWatched` | Count of videos watched |
| `NumberOfQuizzesTaken` | Number of quizzes attempted |
| `QuizScores` | Average quiz score (0–100) |
| `CompletionRate` | Percentage of course content completed |
| `DeviceType` | Device used — `0` = Mobile, `1` = Desktop |
| `CourseCompletion` | **Target variable** — `0` = Not Completed, `1` = Completed |

---

## Project Structure

```
online_course_prediction/
│
├── ml_classification.ipynb        # Full ML pipeline notebook
├── app.py                         # Streamlit web application
├── modelrf.pkl                    # Trained Random Forest model (pickle)
├── online_course_engagement_data.csv  # Raw dataset
├── requirements.txt               # Python dependencies
└── README.md
```

---

## ML Pipeline

The complete machine learning workflow is documented in `ml_classification.ipynb`:

### 1. Data Loading & Validation
- Loaded ~9,000 records from CSV.
- Used first 2,000 rows for faster experimentation; the saved model (`modelrf.pkl`) was trained on this 2,000-row subset.
- Checked unique values, data types, and duplicates.

### 2. Exploratory Data Analysis (EDA)

| Analysis Type | Techniques Used |
|---|---|
| **Univariate** | Histograms, KDE plots, descriptive stats (mean, median, mode, std, skewness) |
| **Bivariate** | Box plots and pie charts per `CourseCategory` |
| **Multivariate** | Correlation heatmap of numerical features |

### 3. Missing Value Handling
The original dataset had no nulls, so nulls were artificially introduced to practise handling:

| Column | % Nulls Introduced | Strategy |
|---|---|---|
| `NumberOfVideosWatched` | 10% | Filled with **mean** |
| `CompletionRate` | 5% | Compared mean / median / mode via KDE plots → filled with **median** |

### 4. Feature Engineering
- `CourseCategory` (categorical string) encoded to numeric using a manual mapping:

  | Category | Encoded Value |
  |---|---|
  | Health | 1 |
  | Arts | 2 |
  | Science | 3 |
  | Programming | 4 |
  | Business | 5 |

- `UserID` was dropped as it carries no predictive value.

### 5. Outlier Detection
- Outliers identified using the **IQR method** (1.5 × IQR rule) and visualised with box plots.

### 6. Train-Test Split & Scaling
- **80 / 20** train-test split (`random_state=123`).
- Features standardised using `StandardScaler`.

### 7. Modelling — 7 Algorithms Compared
Because the target variable (`CourseCompletion`) was **imbalanced**, the **F1 score** was used as the primary evaluation metric instead of accuracy.

---

## Model Comparison

Seven classification algorithms were trained and evaluated:

| # | Algorithm | Notes |
|---|---|---|
| 1 | Logistic Regression | `penalty='l2'`, `class_weight='balanced'` |
| 2 | K-Nearest Neighbours | `n_neighbors=5` |
| 3 | Support Vector Machine | `kernel='rbf'`, `gamma=5` |
| 4 | Naïve Bayes | `GaussianNB` |
| 5 | Decision Tree | `criterion='entropy'`, `max_depth=10` |
| 6 | **Random Forest** ✅ | `n_estimators=100`, `max_depth=10` — **Best model** |
| 7 | XGBoost | `n_estimators=100`, `learning_rate=0.1`, `max_depth=6` |

**Random Forest** achieved the highest F1 scores on both training and test sets and was selected as the final model. It was saved using `pickle` as `modelrf.pkl`.

---

## Web Application

The Streamlit app (`app.py`) provides a user-friendly interface for real-time predictions.

### Features
- Displays a sample dataset preview.
- Accepts the following inputs from the user:

  | Input | Type |
  |---|---|
  | Course Category | Dropdown (Health / Arts / Science / Programming / Business) |
  | Time Spent On Course | Number |
  | Number of Videos Watched | Number |
  | Number of Quizzes Taken | Number |
  | Quiz Scores | Number |
  | Completion Rate | Number |
  | Device Type | Number (0 = Mobile, 1 = Desktop) |

- Loads the saved Random Forest model (`modelrf.pkl`) and predicts course completion.
- Displays the result:
  - ✅ **Course Completed**
  - ❌ **Course Not Completed**

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Just-hari/online_course_prediction.git
cd online_course_prediction

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Run the Streamlit Web App
```bash
streamlit run app.py
```
Then open the local URL shown in the terminal (usually `http://localhost:8501`) in your browser.

### Run the Jupyter Notebook
```bash
jupyter notebook ml_classification.ipynb
```
Execute all cells in order to reproduce the full ML pipeline, including model training and saving.

---

## Technologies Used

| Tool / Library | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib / Seaborn** | Static visualisations |
| **Plotly** | Interactive charts |
| **Scikit-learn** | ML algorithms, preprocessing, evaluation |
| **XGBoost** | Gradient boosting classifier |
| **Pickle** | Model serialisation |
| **Streamlit** | Web application framework |