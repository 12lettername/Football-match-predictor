import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# --- 1. Define Functions ---

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "win"
    elif row["home_score"] < row["away_score"]:
        return "loss"
    else:
        return "draw"

def get_form(team, date, df):
    past_matches = df[df["date"] < date]
    team_matches = past_matches[(past_matches["home_team"] == team) | (past_matches["away_team"] == team)]
    team_matches = team_matches.sort_values(by="date")
    last_5_matches = team_matches.tail(5)
    
    points = 0
    for _, match in last_5_matches.iterrows():
        if match["home_team"] == team:
            if match["home_score"] > match["away_score"]:
                points += 3  
            elif match["home_score"] == match["away_score"]:
                points += 1  
        else:
            if match["away_score"] > match["home_score"]:
                points += 3  
            elif match["away_score"] == match["home_score"]:
                points += 1  
    return points

# --- 2. Define Cached Data & Model Loaders ---

@st.cache_data
def load_data():
    data = requests.get("https://raw.githubusercontent.com/martj42/international_results/master/results.csv").text
    df = pd.read_csv(StringIO(data), parse_dates=["date"])
    df = df[df["tournament"] != "Friendly"]
    df = df[df["date"] >= "2000-01-01"]
    df["result"] = df.apply(get_result, axis=1)
    return df

@st.cache_data
def compute_elo(df, k=32, base=1500):
    elo = {}
    for _, row in df.sort_values("date").iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_elo = elo.get(home, base)
        away_elo = elo.get(away, base)
        
        expected = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        
        if row["result"] == "win":
            actual = 1
        elif row["result"] == "draw":
            actual = 0.5
        else:
            actual = 0
            
        elo[home] = home_elo + k * (actual - expected)
        elo[away] = away_elo + k * ((1 - actual) - (1 - expected))
    return elo

@st.cache_data
def build_features(df, _elo_dict):
    records = []
    for _, row in df.iterrows():
        home_elo = _elo_dict.get(row["home_team"], 1500)
        away_elo = _elo_dict.get(row["away_team"], 1500)
        home_form = get_form(row["home_team"], row["date"], df)
        away_form = get_form(row["away_team"], row["date"], df)
        
        records.append({
            "elo_diff"  : home_elo - away_elo,
            "is_neutral": int(row["neutral"]),
            "home_form" : home_form,
            "away_form" : away_form,
            "form_diff" : home_form - away_form,
            "result"    : row["result"]
        })
    return pd.DataFrame(records)

@st.cache_resource
def train_model(features_df):
    X = features_df[["elo_diff", "is_neutral", "home_form", "away_form", "form_diff"]] 
    y = features_df["result"]                                                      
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# --- 3. Run Pipeline ---

with st.spinner("Loading data and training model... (This will take a minute on the first run)"):
    df = load_data()
    elo = compute_elo(df)
    features_df = build_features(df, elo)
    clf = train_model(features_df)

# --- 4. Build UI ---

st.title("International Football Match Predictor ⚽")

teams = sorted(list(elo.keys()))

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Select Home Team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
with col2:
    away_team = st.selectbox("Select Away Team", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)

is_neutral = st.checkbox("Match played at a neutral venue?", value=False)

if st.button("Predict"):
    if home_team == away_team:
        st.warning("Please select two different teams!")
    else:
        # 1. Get Elo
        home_elo = elo.get(home_team, 1500)
        away_elo = elo.get(away_team, 1500)
        
        # 2. Get Form (Using today's date for a future match)
        today = pd.Timestamp.today()
        home_form = get_form(home_team, today, df)
        away_form = get_form(away_team, today, df)
        
        # 3. Build single-row DataFrame exactly as the model expects
        match_features = pd.DataFrame([{
            "elo_diff"  : home_elo - away_elo,
            "is_neutral": int(is_neutral),
            "home_form" : home_form,
            "away_form" : away_form,
            "form_diff" : home_form - away_form
        }])
        
        # 4. Predict
        probs = clf.predict_proba(match_features)[0]
        classes = clf.classes_  # e.g., ['draw', 'loss', 'win']
        
        # 5. Display the results clearly
        st.subheader(f"Results: {home_team} vs {away_team}")
        
        # Create a nice layout to show the probabilities
        for cls, prob in zip(classes, probs):
            # Format the text so 'loss' means 'away win', 'win' means 'home win', etc.
            if cls == "win":
                label = f"{home_team} Win"
            elif cls == "loss":
                label = f"{away_team} Win"
            else:
                label = "Draw"
                
            st.write(f"**{label}:** {prob * 100:.1f}%")