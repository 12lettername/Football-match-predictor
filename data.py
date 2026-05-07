import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

data = requests.get("https://raw.githubusercontent.com/martj42/international_results/master/results.csv").text

df = pd.read_csv(StringIO(data), parse_dates=["date"])

df = df[df["tournament"] != "Friendly"]
df = df[df["date"] >= "2000-01-01"]

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "win"
    elif row["home_score"] < row["away_score"]:
        return "loss"
    else:
        return "draw"

df["result"] = df.apply(get_result, axis = 1)

def compute_elo(df, k=32, base=1500):
    elo = {}

    for _, row in df.sort_values("date").iterrows():
        home = row["home_team"]
        away = row["away_team"]

        
        home_elo = elo.get(home, base)
        away_elo = elo.get(away, base)

        
        expected = 1 / (1 + 10 ** ((away_elo- home_elo) / 400))

        
        if row["result"] == "win":
            actual = 1
        elif row["result"] == "draw":
            actual = 0.5
        else:
            actual = 0

        elo[home] = home_elo + k * (actual - expected)
        elo[away] = away_elo + k * ((1 - actual) - (1 - expected))

    return elo

# print(df.shape)
# print(df["result"].value_counts())
# print(df.head(10))

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

elo = compute_elo(df)
# top10 = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:10]
# for team, rating in top10:
#     print(f"{team}: {rating:.0f}")
records = []

for _, row in df.iterrows():
    
    home_elo = elo.get(row["home_team"], 1500)
    away_elo = elo.get(row["away_team"], 1500)
    
    
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
features_df = pd.DataFrame(records)
# print(features_df.tail())



X = features_df[["elo_diff", "is_neutral", "home_form", "away_form", "form_diff"]] 
y = features_df["result"]                                                             

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_pred, y_test))
feature_names = ["elo_diff", "is_neutral", "home_form", "away_form", "form_diff"]
importances = clf.feature_importances_

for name, score in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.3f}")