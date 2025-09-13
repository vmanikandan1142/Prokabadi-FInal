import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import base64

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Team Analyzer", layout="wide")

# -------------------------------
# Add Background Image Function
# -------------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                              url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
        }}
        h1, h2, h3, h4, h5, h6, p, div, label, span {{
            color: white !important;
        }}
        .stDataFrame, .stTable {{
            color: black !important;
        }}

        /* ---------- Make Tabs Bigger ---------- */
        .stTabs [role="tablist"] button {{
            font-size: 20px !important;       /* bigger text */
            padding: 12px 30px !important;    /* bigger clickable area */
            margin: 5px !important;           /* spacing between tabs */
            min-width: 150px !important;      /* minimum width of each tab */
        }}
        .stTabs [role="tablist"] button[aria-selected="true"] {{
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 10px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("bg.png")

# -------------------------------
# Load Data & Models
# -------------------------------
df = pd.read_csv("teamStats_needed.csv")
with open("cart_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else model.get_booster().feature_names

# -------------------------------
# Global Functions
# -------------------------------
def preprocess_team(team_name):
    team_data = df[df['team_name'] == team_name].mean(numeric_only=True).to_frame().T
    one_hot = pd.get_dummies(df[['team_name', 'stats_status']], columns=['stats_status'])
    one_hot_team = one_hot[one_hot['team_name'] == team_name].drop(columns='team_name').mean().to_frame().T
    combined = pd.concat([team_data, one_hot_team], axis=1)
    combined = combined.reindex(columns=model_features, fill_value=0)
    return combined

def predict_winner(t1, t2):
    team1_input = preprocess_team(t1)
    team2_input = preprocess_team(t2)
    team1_pred = model.predict(team1_input)[0]
    team2_pred = model.predict(team2_input)[0]
    if abs(team1_pred - team2_pred) < 0.1:
        return random.choice([t1, t2])
    return t1 if team1_pred > team2_pred else t2

# -------------------------------
# Custom Title
# -------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size:40px;
        text-align:center;
        color:white;
        background-color:#4CAF50;
        padding:20px;
        border-radius:12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="main-title">🏆 Team Stats Dashboard & Predictor</div>', unsafe_allow_html=True)

# -------------------------------
# Tabs Navigation
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🏠 Dashboard", "📊 Team Plot", "🔮 Tournament Prediction", "⚔ Rivalry Comparison"]
)

# -------------------------------
# Dashboard Tab
# -------------------------------
with tab1:
    st.subheader("📈 Team Win Percentages")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Teams", df['team_name'].nunique())
    col2.metric("Matches Played", int(df['total_matches'].sum()))
    col3.metric("Total Wins", int(df['wins'].sum()))

    win_pct = df.groupby("team_name")[["wins", "total_matches"]].sum()
    win_pct["Win %"] = (win_pct["wins"] / win_pct["total_matches"]) * 100
    win_pct = win_pct.sort_values("Win %")

    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    ax.barh(win_pct.index, win_pct["Win %"], color='skyblue')
    ax.set_title("Win Percentage by Team", color="white")
    ax.set_xlabel("Win %", color="white")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    for i, v in enumerate(win_pct["Win %"]):
        ax.text(v + 1, i, f"{v:.1f}%", color="white", va='center', fontsize=9)

    st.pyplot(fig)

# -------------------------------
# Team Plot Tab
# -------------------------------
with tab2:
    st.subheader("📊 Team Attribute Performance")
    teams = df['team_name'].unique()
    team = st.selectbox("Select Team", teams, key="team_select")
    attribute = st.selectbox(
        "Select Attribute",
        [col for col in df.columns if df[col].dtype != 'object' and col not in ['wins', 'total_matches']],
        key="attr_select"
    )
    plot_type = st.radio("Select Plot Type", ["Line", "Bar", "Scatter"], key="plot_type")

    team_data = df[df["team_name"] == team]
    x = np.arange(len(team_data))
    y = team_data[attribute].values.astype(float)

    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    if plot_type == "Line":
        ax.plot(x, y, marker='o', linestyle='-', color='purple')
        for i, val in enumerate(y):
            ax.text(i, val, f"{val:.1f}", color="white", fontsize=8, ha='center', va='bottom')
    elif plot_type == "Bar":
        ax.bar(x, y, color='orange')
        for i, val in enumerate(y):
            ax.text(i, val, f"{val:.1f}", color="white", fontsize=8, ha='center', va='bottom')
    elif plot_type == "Scatter":
        ax.scatter(x, y, color='green')
        for i, val in enumerate(y):
            ax.text(i, val, f"{val:.1f}", color="white", fontsize=8, ha='center', va='bottom')

    ax.set_title(f"{team} - {attribute} Trend", color="white")
    ax.set_ylabel(attribute, color="white")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    st.pyplot(fig)

# -------------------------------
# Tournament Prediction Tab
# -------------------------------
with tab3:
    st.subheader("🔮 Simulate Tournament")

    win_pct = df.groupby("team_name")[["wins", "total_matches"]].sum()
    win_pct["Win %"] = (win_pct["wins"] / win_pct["total_matches"]) * 100
    top_8_teams = win_pct["Win %"].sort_values(ascending=False).head(8).index.tolist()

    if st.button("▶ Run Tournament", key="tournament_btn"):
        st.markdown("### 🏃 Matches")
        points = {team: 0 for team in top_8_teams}

        for i in range(len(top_8_teams)):
            for j in range(i + 1, len(top_8_teams)):
                t1, t2 = top_8_teams[i], top_8_teams[j]
                winner = predict_winner(t1, t2)
                points[winner] += 2
                st.write(f"{t1} vs {t2} → **{winner} wins**")

        st.markdown("### 📊 Points Table")
        points_df = pd.DataFrame.from_dict(points, orient='index', columns=['Points']).sort_values(by='Points', ascending=False)
        points_df.index.name = 'Team'
        st.dataframe(points_df)

        top4 = points_df.head(4).index.tolist()
        st.markdown("### 💪 Semi Finals")
        semi_matchups = [(top4[0], top4[3]), (top4[1], top4[2])]
        semi_winners = []
        for i, (t1, t2) in enumerate(semi_matchups):
            winner = predict_winner(t1, t2)
            st.markdown(f"**Semi Final {i+1}: {t1} vs {t2} → 🌟 **{winner} wins** 🌟")
            semi_winners.append(winner)

        st.markdown("### 🏆 Final Match")
        final_team1, final_team2 = semi_winners[0], semi_winners[1]
        final_winner = predict_winner(final_team1, final_team2)
        st.markdown(f"### 🏆 Final: {final_team1} vs {final_team2}")
        st.markdown(
            f"<div style='background-color: #4CAF50; padding: 20px; font-size: 28px; text-align: center; color: white;'>🏆 **{final_winner} is the Champion!** 🏆</div>",
            unsafe_allow_html=True
        )

# -------------------------------
# Rivalry Comparison Tab
# -------------------------------
with tab4:
    st.subheader("⚔ Compare Two Teams")
    teams = df["team_name"].unique()
    team1 = st.selectbox("Select First Team", teams, key="rival_team1")
    team2 = st.selectbox("Select Second Team", [t for t in teams if t != team1], key="rival_team2")

    winner = predict_winner(team1, team2)
    
    # Display the result in an extra large font
    st.markdown(
        f"<h2 style='text-align: center; color: #2E86C1;'>🔮 Based on the analysis, <b>{winner}</b> is likely to win the rivalry!</h1>", 
        unsafe_allow_html=True
    )
