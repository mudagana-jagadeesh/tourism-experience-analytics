"""
Tourism Experience Analytics — Streamlit App
=============================================
Lets a user enter their profile (location, planned visit context) and get:
  1. A predicted Visit Mode (Business / Couples / Family / Friends / Solo)
  2. Personalized attraction recommendations (hybrid collaborative + content-based)
  3. Visualizations of popular attractions, top regions, and user segments

Run with:  streamlit run app.py
Requires the ./artifacts/ folder produced by Tourism_Experience_Analytics.ipynb
(rf_regressor.pkl, xgb_classifier.pkl / rf_classifier.pkl, label_encoders.pkl,
svd_model.pkl, pred_ratings.pkl, item_similarity.pkl, attraction_lookup.pkl, master.pkl)
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="🏝️", layout="wide")

ARTIFACT_DIR = "artifacts"


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(os.path.join(ARTIFACT_DIR, "label_encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)

    # Random Forest is the production classifier per the project brief
    clf_path = os.path.join(ARTIFACT_DIR, "rf_classifier.pkl")
    if not os.path.exists(clf_path):
        clf_path = os.path.join(ARTIFACT_DIR, "xgb_classifier.pkl")
    with open(clf_path, "rb") as f:
        classifier = pickle.load(f)

    with open(os.path.join(ARTIFACT_DIR, "rf_regressor.pkl"), "rb") as f:
        regressor = pickle.load(f)

    with open(os.path.join(ARTIFACT_DIR, "svd_model.pkl"), "rb") as f:
        svd = pickle.load(f)

    pred_ratings = pd.read_pickle(os.path.join(ARTIFACT_DIR, "pred_ratings.pkl"))
    item_similarity = pd.read_pickle(os.path.join(ARTIFACT_DIR, "item_similarity.pkl"))
    attraction_lookup = pd.read_pickle(os.path.join(ARTIFACT_DIR, "attraction_lookup.pkl"))
    master = pd.read_pickle(os.path.join(ARTIFACT_DIR, "master.pkl"))

    return {
        "encoders": encoders,
        "classifier": classifier,
        "regressor": regressor,
        "svd": svd,
        "pred_ratings": pred_ratings,
        "item_similarity": item_similarity,
        "attraction_lookup": attraction_lookup,
        "master": master,
    }


def safe_label_encode(le, value):
    """Encode a category value, falling back to the first known class if unseen."""
    try:
        return le.transform([str(value)])[0]
    except ValueError:
        return le.transform([le.classes_[0]])[0]


# ----------------------------------------------------------------------
# Recommendation helpers
# ----------------------------------------------------------------------
def recommend_for_existing_user(art, user_id, n=5, cf_weight=0.6):
    pred_ratings = art["pred_ratings"]
    master = art["master"]
    item_similarity = art["item_similarity"]
    attraction_lookup = art["attraction_lookup"]

    if user_id not in pred_ratings.index:
        return None

    already_visited = master.loc[master["UserId"] == user_id, "AttractionId"].unique()
    cf_scores = pred_ratings.loc[user_id].drop(labels=already_visited, errors="ignore")
    denom = (cf_scores.max() - cf_scores.min()) or 1e-9
    cf_scores = (cf_scores - cf_scores.min()) / denom

    liked = master[(master["UserId"] == user_id) & (master["Rating"] >= 4)]["AttractionId"].unique()
    if len(liked):
        cb_scores = item_similarity[liked].mean(axis=1).drop(labels=already_visited, errors="ignore")
        denom_cb = (cb_scores.max() - cb_scores.min()) or 1e-9
        cb_scores = (cb_scores - cb_scores.min()) / denom_cb
    else:
        cb_scores = pd.Series(0, index=cf_scores.index)

    combined = cf_weight * cf_scores + (1 - cf_weight) * cb_scores.reindex(cf_scores.index).fillna(0)
    top_n = combined.sort_values(ascending=False).head(n)
    return attraction_lookup.loc[top_n.index].assign(Score=top_n.values.round(3))


def recommend_cold_start(art, preferred_type, n=5):
    """Content-based fallback for a brand-new user with no visit history:
    recommend the highest-rated, most-visited attractions of their preferred type
    (or overall best attractions if no preference given)."""
    master = art["master"]
    attraction_lookup = art["attraction_lookup"]

    pool = master if preferred_type in (None, "Any / No preference") else master[master["AttractionType"] == preferred_type]
    stats = pool.groupby("AttractionId").agg(AvgRating=("Rating", "mean"), Visits=("TransactionId", "count"))
    stats = stats[stats["Visits"] >= 20]  # avoid tiny-sample noise
    top_n = stats.sort_values(["AvgRating", "Visits"], ascending=False).head(n)
    return attraction_lookup.loc[top_n.index].assign(
        AvgRating=top_n["AvgRating"].values.round(2), Visits=top_n["Visits"].values
    )


# ----------------------------------------------------------------------
# App layout
# ----------------------------------------------------------------------
st.title("🏝️ Tourism Experience Analytics")
st.caption("Personalized visit-mode prediction and attraction recommendations, powered by ML trained on 52,930 real visit transactions.")

art = load_artifacts()
master = art["master"]
encoders = art["encoders"]["features"]
target_encoder = art["encoders"]["target"]

tab1, tab2, tab3 = st.tabs(["🔮 Predict & Recommend", "📊 Tourism Analytics", "ℹ️ About"])

# ============================ TAB 1: Predict & Recommend ============================
with tab1:
    st.subheader("Tell us about yourself")

    col1, col2 = st.columns(2)
    with col1:
        continent_opts = sorted(encoders["UserContinent"].classes_)
        continent = st.selectbox("Continent", continent_opts, index=continent_opts.index("Asia") if "Asia" in continent_opts else 0)

        region_opts_all = sorted(encoders["UserRegion"].classes_)
        region_candidates = master.loc[master["UserContinent"] == continent, "UserRegion"].dropna().unique().tolist()
        region_opts = sorted(region_candidates) if region_candidates else region_opts_all
        region = st.selectbox("Region", region_opts)

        country_candidates = master.loc[master["UserRegion"] == region, "UserCountry"].dropna().unique().tolist()
        country_opts = sorted(country_candidates) if country_candidates else sorted(encoders["UserCountry"].classes_)
        country = st.selectbox("Country", country_opts)

    with col2:
        visit_year = st.number_input("Planned Visit Year", min_value=2013, max_value=2030, value=2026)
        visit_month = st.selectbox("Planned Visit Month", list(range(1, 13)), index=6,
                                    format_func=lambda m: pd.Timestamp(2024, m, 1).strftime('%B'))
        attraction_type_opts = sorted(encoders["AttractionType"].classes_)
        preferred_type = st.selectbox("Preferred Attraction Type (for recommendations)",
                                       ["Any / No preference"] + attraction_type_opts)

    existing_user_id = st.number_input(
        "Returning visitor? Enter your User ID for personalized (collaborative-filtering) recommendations. "
        "Leave at 0 for a new-visitor experience.",
        min_value=0, value=0, step=1,
    )

    if st.button("Get My Prediction & Recommendations", type="primary"):
        # --- Visit Mode Prediction ---
        attraction_continent_opts = encoders["AttractionContinent"].classes_
        default_attr_continent = attraction_continent_opts[0]

        clf_input = pd.DataFrame([{
            "VisitYear": visit_year,
            "VisitMonth": visit_month,
            "AttractionTypeId": 0,  # unknown at prediction time; model relies mainly on geography/history
            "UserContinent_enc": safe_label_encode(encoders["UserContinent"], continent),
            "UserRegion_enc": safe_label_encode(encoders["UserRegion"], region),
            "UserCountry_enc": safe_label_encode(encoders["UserCountry"], country),
            "AttractionType_enc": safe_label_encode(
                encoders["AttractionType"], preferred_type if preferred_type != "Any / No preference" else encoders["AttractionType"].classes_[0]
            ),
            "AttractionContinent_enc": safe_label_encode(encoders["AttractionContinent"], default_attr_continent),
            "UserVisitCount": 1,
        }])

        pred_mode = art["classifier"].predict(clf_input)
        # XGBoost returns an encoded int; RandomForest returns the label directly
        if isinstance(pred_mode[0], (np.integer, int)):
            pred_mode_label = target_encoder.inverse_transform(pred_mode)[0]
        else:
            pred_mode_label = pred_mode[0]

        st.success(f"### Predicted Visit Mode: **{pred_mode_label}**")

        # --- Recommendations ---
        st.subheader("Recommended Attractions For You")
        recs = None
        if existing_user_id and existing_user_id > 0:
            recs = recommend_for_existing_user(art, existing_user_id, n=5)
            if recs is None:
                st.info(f"User ID {existing_user_id} not found in history — showing new-visitor recommendations instead.")

        if recs is None:
            recs = recommend_cold_start(art, preferred_type, n=5)

        st.dataframe(recs[["Attraction", "AttractionType"] + [c for c in recs.columns if c not in ("Attraction", "AttractionType")]],
                     use_container_width=True)

# ============================ TAB 2: Tourism Analytics ============================
with tab2:
    st.subheader("Popular Attractions & Regional Trends")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 10 Attractions by Visit Volume**")
        top10 = master["Attraction"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 5))
        top10.sort_values().plot(kind="barh", ax=ax, color="mediumpurple")
        ax.set_xlabel("Number of Visits")
        st.pyplot(fig)

    with c2:
        st.markdown("**Average Rating by Attraction Type**")
        avg_type = master.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        avg_type.plot(kind="barh", ax=ax2, color="teal")
        ax2.invert_yaxis()
        ax2.set_xlabel("Average Rating")
        st.pyplot(fig2)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**User Distribution by Continent**")
        cont = master["UserContinent"].value_counts()
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        cont.plot(kind="bar", ax=ax3, color="goldenrod")
        plt.xticks(rotation=30)
        st.pyplot(fig3)

    with c4:
        st.markdown("**Visit Mode Distribution (User Segments)**")
        vm = master["VisitModeName"].value_counts()
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        vm.plot(kind="bar", ax=ax4, color="steelblue")
        plt.xticks(rotation=0)
        st.pyplot(fig4)

    st.markdown("**Seasonality — Visits by Month**")
    monthly = master.groupby("VisitMonth").size()
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    monthly.plot(kind="bar", ax=ax5, color="seagreen")
    ax5.set_xlabel("Month")
    st.pyplot(fig5)

# ============================ TAB 3: About ============================
with tab3:
    st.markdown("""
### About this app
This app is the deployment layer for the **Tourism Experience Analytics** project. It combines three ML components trained in `Tourism_Experience_Analytics.ipynb`:

- **Classification model** (XGBoost) — predicts a visitor's likely Visit Mode from their location and visit context.
- **Recommendation system** — hybrid collaborative filtering (SVD on the user-item rating matrix) + content-based filtering (attraction-type similarity), with a cold-start fallback for brand-new visitors.
- **Analytics dashboard** — descriptive visualizations of attraction popularity, satisfaction, and visitor demographics drawn directly from the cleaned transaction dataset.

**Data:** 52,930 visit transactions across 33,530 users and 30 tourist attractions (Transaction, User, City, Country, Region, Continent, Item, Type, and Mode tables).

**Note:** predictions are only as good as the underlying signal — see the notebook's EDA and hypothesis-testing sections for an honest discussion of how much demographic/categorical data actually explains rating and visit mode in this dataset.
""")
