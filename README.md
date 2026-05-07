# International Football Match Predictor ⚽

A machine learning web application built with Python and Streamlit that predicts the outcomes of international football (soccer) matches. The model calculates win/draw/loss probabilities based on historical data, dynamic Elo ratings, and recent team form.

## 🌟 Features

- **Dynamic Elo Ratings:** Calculates a custom Elo rating for every international team based on match results since the year 2000.
- **Form Tracking:** Analyzes the last 5 matches for any given team to calculate their current momentum (Points: 3 for Win, 1 for Draw, 0 for Loss).
- **Machine Learning:** Uses a Scikit-Learn `GradientBoostingClassifier` trained on engineered features (Elo difference, form difference, and neutral venue status).
- **Interactive UI:** Built with Streamlit, allowing users to easily select two teams and see the exact probability of a Home Win, Away Win, or Draw.
- **Automated Data Fetching:** Automatically pulls the latest historical match dataset from GitHub on launch.

## 🛠️ Technologies Used

- **Python 3**
- **Streamlit** (Web framework)
- **Pandas** (Data manipulation)
- **Scikit-Learn** (Machine learning model)
- **Requests** (Fetching remote data)

## 💻 Installation and Setup

1. **Clone or download this repository** to your local machine.
2. **Ensure you have Python installed** (Python 3.8 or higher is recommended).
3. **Install the required libraries** by opening your terminal in the project folder and running:
   ```bash
   pip install streamlit pandas scikit-learn requests
   ```
