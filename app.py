import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# AI imports
from textblob import TextBlob
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ------------------------------------------------------------
# App config
# ------------------------------------------------------------
st.set_page_config(page_title="Student Behaviour Tracker", layout="wide")
st.title("Student Behaviour Tracker")

# ------------------------------------------------------------
# Required columns and score fields
# ------------------------------------------------------------
required_columns = [
    "Student", "Year Group", "Date", "Behaviour", "Home Life", "Eating Habits",
    "Disabilities", "Interventions", "Safeguarding", "Social", "Comments", "Intervention Details"
]

score_fields = [
    "Behaviour", "Home Life", "Eating Habits",
    "Disabilities", "Interventions", "Safeguarding", "Social"
]

# ------------------------------------------------------------
# Utilities: Rule-based scoring and risk
# ------------------------------------------------------------
def categorize_risk(score: float) -> str:
    if score >= 40:
        return "Stable"
    elif score >= 25:
        return "Monitor"
    else:
        return "High Risk"

def ensure_scores_and_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Coerce numeric
    for col in score_fields:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Total score
    df["Total Score"] = df[score_fields].sum(axis=1)
    # Risk category
    df["Risk Category"] = df["Total Score"].apply(categorize_risk)
    return df

# ------------------------------------------------------------
# AI: Sentiment and features
# ------------------------------------------------------------
def comment_sentiment(text) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    try:
        return float(TextBlob(text).sentiment.polarity)  # -1 to +1
    except Exception:
        return 0.0

def build_features(df: pd.DataFrame):
    df = df.copy()
    # Sentiment feature
    df["Comment Sentiment"] = df["Comments"].apply(comment_sentiment)
    # Normalize types
    for col in score_fields:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Year Group"] = df["Year Group"].astype(str)
    # Features and labels
    X = df[score_fields + ["Comment Sentiment", "Year Group"]].copy()
    y = df["Risk Category"].astype(str)
    return X, y

# ------------------------------------------------------------
# AI: Train predictive model
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def train_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state))
    ])

    # Stratify if possible
    stratify = y if len(y.unique()) > 1 and min(y.value_counts()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )
    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    return model, report

# ------------------------------------------------------------
# AI: Predict with probabilities
# ------------------------------------------------------------
def predict_with_proba(model, X: pd.DataFrame):
    try:
        proba = model.predict_proba(X)
        preds = model.predict(X)
        classes = model.named_steps["clf"].classes_
        return preds, proba, classes
    except Exception:
        preds = model.predict(X)
        classes = np.unique(preds)
        # Pseudo-probabilities
        proba = np.zeros((len(preds), len(classes)))
        for i, p in enumerate(preds):
            proba[i, np.where(classes == p)[0][0]] = 1.0
        return preds, proba, classes

# ------------------------------------------------------------
# AI: Anomaly detection
# ------------------------------------------------------------
def detect_anomalies(X: pd.DataFrame):
    # Use only numeric features for isolation forest
    numeric_X = X.select_dtypes(include=[np.number])
    if numeric_X.empty or len(numeric_X) < 5:
        # Not enough data to detect anomalies meaningfully
        return np.zeros(len(X), dtype=int)
    iso = IsolationForest(contamination=0.1, random_state=42)
    preds = iso.fit_predict(numeric_X)  # -1 anomaly, 1 normal
    return preds

# ------------------------------------------------------------
# Data load/upload and seed
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Student Behaviour CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if set(required_columns).issubset(df.columns):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        st.session_state.data = df.copy()
        st.success("CSV uploaded successfully.")
    else:
        st.error("Uploaded CSV is missing required columns.")
        st.session_state.data = pd.DataFrame(columns=required_columns)
else:
    if "data" not in st.session_state:
        @st.cache_data
        def load_data():
            return pd.DataFrame(columns=required_columns)
        st.session_state.data = load_data()

        # Simulated entry
        simulated_entry = {
            "Student": "Jordan M.",
            "Year Group": "Year 9",
            "Date": datetime.today(),
            "Behaviour": 4,
            "Home Life": 6,
            "Eating Habits": 5,
            "Disabilities": 7,
            "Interventions": 3,
            "Safeguarding": 8,
            "Social": 4,
            "Comments": "Jordan has been quiet and disengaged. Referred to mentoring but hasn’t attended yet.",
            "Intervention Details": "Mentoring referral made; awaiting first session."
        }
        st.session_state.data = pd.concat(
            [st.session_state.data, pd.DataFrame([simulated_entry])],
            ignore_index=True
        )

# Ensure rule-based totals and risk exist
data = ensure_scores_and_risk(st.session_state.data)
st.session_state.data = data

# Flag invalid dates
invalid_dates = data[data["Date"].isna()]
if not invalid_dates.empty:
    st.warning(f"{len(invalid_dates)} entries have invalid dates.")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Add Entry", "Overview", "Visuals", "Interventions",
    "Comments", "Summary", "Search", "Scoring Guide", "AI Insights"
])

# ------------------------------------------------------------
# Tab 1: Add Entry
# ------------------------------------------------------------
with tab1:
    st.header("Add New Behaviour Entry")
    with st.form("entry_form"):
        student = st.text_input("Student Name")
        year_group = st.selectbox("Year Group", [f"Year {i}" for i in range(7, 14)])
        date = st.date_input("Date", value=datetime.today())
        behaviour = st.slider("Behaviour", 1, 10, 5)
        home = st.slider("Home Life", 1, 10, 5)
        eating = st.slider("Eating Habits", 1, 10, 5)
        disability = st.slider("Disabilities", 1, 10, 5)
        intervention = st.slider("Attending Interventions", 1, 10, 5)  # label aligned
        safeguarding = st.slider("Safeguarding Issues", 1, 10, 5)
        social = st.slider("Social", 1, 10, 5)
        comments = st.text_area("Comments")
        intervention_details = st.text_area("Describe Current Interventions")
        submitted = st.form_submit_button("Submit Entry")

        if submitted:
            if not student.strip():
                st.error("Student name is required.")
            else:
                new_row = {
                    "Student": student,
                    "Year Group": year_group,
                    "Date": pd.to_datetime(date),
                    "Behaviour": behaviour,
                    "Home Life": home,
                    "Eating Habits": eating,
                    "Disabilities": disability,
                    "Interventions": intervention,
                    "Safeguarding": safeguarding,
                    "Social": social,
                    "Comments": comments,
                    "Intervention Details": intervention_details
                }
                st.session_state.data = pd.concat(
                    [st.session_state.data, pd.DataFrame([new_row])],
                    ignore_index=True
                )
                # Recompute scores and risk
                st.session_state.data = ensure_scores_and_risk(st.session_state.data)
                st.success("Entry added successfully.")

# ------------------------------------------------------------
# Tab 2: Overview
# ------------------------------------------------------------
with tab2:
    st.header("Student Overview")
    student_list = st.session_state.data["Student"].dropna().unique()
    if len(student_list) == 0:
        st.info("No students available.")
    else:
        selected_student = st.selectbox("Select Student", student_list, key="overview_student")
        student_data = st.session_state.data[st.session_state.data["Student"] == selected_student]

        if not student_data.empty:
            latest = student_data.sort_values("Date", ascending=False).iloc[0]
            avg_score = student_data["Total Score"].mean()
            status = categorize_risk(avg_score)

            st.metric("Latest Behaviour Score", latest["Total Score"])
            st.metric("Average Behaviour Score", round(avg_score, 1))
            st.metric("Risk Category (avg)", status)
            st.write("Latest Entry:")
            st.dataframe(student_data.sort_values("Date", ascending=False).head(1))

# ------------------------------------------------------------
# Tab 3: Visuals
# ------------------------------------------------------------
with tab3:
    st.header("Behaviour Trends")
    if not st.session_state.data.empty:
        students = st.multiselect("Select Students", st.session_state.data["Student"].dropna().unique())
        if students:
            filtered = st.session_state.data[st.session_state.data["Student"].isin(students)].copy()
            filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")
            chart_data = filtered.pivot_table(
                index=["Date", "Year Group"],
                columns="Student",
                values="Total Score",
                aggfunc="mean"
            ).reset_index()
            chart_data["Date_Label"] = chart_data["Date"].dt.strftime("%Y-%m-%d") + " (" + chart_data["Year Group"] + ")"
            chart_data.set_index("Date_Label", inplace=True)
            st.line_chart(chart_data.drop(columns=["Date", "Year Group"]))
        else:
            st.info("Select one or more students to view behaviour trends.")
    else:
        st.info("No data available.")

# ------------------------------------------------------------
# Tab 4: Interventions
# ------------------------------------------------------------
with tab4:
    st.header("Intervention Log")
    st.dataframe(st.session_state.data[[
        "Student", "Year Group", "Date", "Interventions", "Intervention Details", "Comments"
    ]])

# ------------------------------------------------------------
# Tab 5: Comments
# ------------------------------------------------------------
with tab5:
    st.header("Comments and Notes")
    st.dataframe(st.session_state.data[["Student", "Year Group", "Date", "Comments"]])

# ------------------------------------------------------------
# Tab 6: Summary
# ------------------------------------------------------------
with tab6:
    st.header("Student Summary")
    student = st.selectbox("Select Student for Summary", st.session_state.data["Student"].dropna().unique(), key="summary_student")
    st.dataframe(st.session_state.data[st.session_state.data["Student"] == student])

# ------------------------------------------------------------
# Tab 7: Search
# ------------------------------------------------------------
with tab7:
    st.header("Search Entries")
    query = st.text_input("Search by student name, year group, or keyword")
    if query:
        # Efficient row-wise contains
        mask = st.session_state.data.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)
        results = st.session_state.data[mask]
        st.dataframe(results)

# ------------------------------------------------------------
# Tab 8: Scoring Guide and Downloads
# ------------------------------------------------------------
with tab8:
    st.header("Behaviour Scoring System")

    st.markdown("""
    The Student Behaviour Tracker uses a structured scoring framework across seven behavioural indicators. Each score ranges from **1 (high concern)** to **10 (no concern)**.

    **Scoring Philosophy**
    - 1–3: Significant concern. Urgent support may be needed.
    - 4–6: Moderate concern. Monitor and consider interventions.
    - 7–8: Generally stable. Occasional challenges.
    - 9–10: Strong behavioural indicators. No major concerns.

    **Categories Explained**
    - Behaviour: Engagement, conduct, and classroom interactions.
    - Home Life: Stability, support, and home environment.
    - Eating Habits: Nutrition, appetite, and food access.
    - Disabilities: Impact of physical, cognitive, or learning challenges.
    - Attending Interventions: Participation in support programs or services.
    - Safeguarding Issues: Exposure to risk, neglect, or harm.
    - Social: Peer relationships, isolation, and social confidence.

    **Total Score**
    Sum of all category scores (maximum 70). Reflects overall behavioural status.

    **Predicted Outcome**
    | Total Score Range | Risk Category | Recommended Action |
    |-------------------|---------------|---------------------|
    | 0–24              | High Risk     | Immediate support and review |
    | 25–39             | Monitor       | Regular check-ins and possible intervention |
    | 40–70             | Stable        | No immediate concern; continue observation |
    """)

    if not st.session_state.data.empty:
        # Download All Student Data
        st.subheader("Download All Student Data")
        csv = st.session_state.data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='student_behaviour_data.csv',
            mime='text/csv'
        )

        # Download Individual Student Data
        st.subheader("Download Individual Student Data")
        student_names = sorted(st.session_state.data["Student"].dropna().unique())
        selected_student = st.selectbox("Select Student for Export", student_names, key="download_student")

        student_df = st.session_state.data[st.session_state.data["Student"] == selected_student]
        student_csv = student_df.to_csv(index=False).encode("utf-8")
        filename = f"{selected_student.replace(' ', '_').lower()}_data.csv"

        st.download_button(
            label=f"Download {selected_student}'s Data",
            data=student_csv,
            file_name=filename,
            mime="text/csv"
        )
    else:
        st.info("No data available to download.")

# ------------------------------------------------------------
# Tab 9: AI Insights
# ------------------------------------------------------------
with tab9:
    st.header("AI Insights")
    st.caption("AI predictions complement, not replace, professional judgment. Use them to inform discussion and monitoring.")

    data = st.session_state.data.copy()
    if data.empty:
        st.info("No data available. Add entries or upload a CSV to enable AI insights.")
    else:
        # Build features
        X, y = build_features(data)

        # Class balance check
        class_counts = y.value_counts()
        if (class_counts < 2).any() or len(class_counts) < 2:
            st.warning("Not enough labeled diversity to train a predictive model. Add more entries across risk categories.")
        else:
            # Controls
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                retrain = st.button("Train / Retrain AI model")
            with col2:
                save_model = st.checkbox("Save model to disk (student_behaviour_model.joblib)")
            with col3:
                load_model = st.button("Load saved model")

            # Load or train
            model = st.session_state.get("ai_model", None)
            if load_model:
                try:
                    model = joblib.load("student_behaviour_model.joblib")
                    st.session_state.ai_model = model
                    st.success("Loaded saved model.")
                except Exception as e:
                    st.error(f"Could not load model: {e}")

            if retrain or model is None:
                with st.spinner("Training AI model..."):
                    model, report = train_model(X, y)
                    st.session_state.ai_model = model
                st.success("Model trained.")
                st.write("Validation report (held-out set):")
                st.json(report)

            # Inference
            model = st.session_state.ai_model
            preds, proba, classes = predict_with_proba(model, X)

            # Attach outputs
            data["Comment Sentiment"] = data["Comments"].apply(comment_sentiment)
            data["AI Predicted Risk"] = preds
            data["AI Confidence"] = np.round(proba.max(axis=1), 3)

            # Anomaly detection
            anomaly_preds = detect_anomalies(X)
            data["Anomaly Flag"] = anomaly_preds  # -1 anomaly, 1 normal, 0 unknown

            # Comparison view
            st.subheader("Rule-based vs AI predictions")
            comparison_cols = [
                "Student", "Year Group", "Date", "Total Score",
                "Risk Category", "AI Predicted Risk", "AI Confidence",
                "Comment Sentiment", "Anomaly Flag"
            ]
            available_cols = [c for c in comparison_cols if c in data.columns]
            st.dataframe(data[available_cols].sort_values("Date", ascending=False))

            # Mismatches
            st.subheader("Mismatches to review")
            mismatches = data[data["Risk Category"] != data["AI Predicted Risk"]]
            if mismatches.empty:
                st.info("No mismatches at the moment.")
            else:
                st.dataframe(mismatches[available_cols].sort_values("AI Confidence", ascending=False))

            # Per-student AI insight
            st.subheader("Per-student AI insight")
            student_names = sorted(data["Student"].dropna().unique())
            pick = st.selectbox("Select student", student_names, key="ai_student_pick")
            sdata = data[data["Student"] == pick].sort_values("Date", ascending=False)
            if not sdata.empty:
                st.metric("Latest rule-based risk", sdata.iloc[0]["Risk Category"])
                st.metric("Latest AI predicted risk", sdata.iloc[0]["AI Predicted Risk"])
                st.metric("AI confidence", sdata.iloc[0]["AI Confidence"])
                st.metric("Latest comment sentiment", np.round(sdata.iloc[0]["Comment Sentiment"], 3))

            # Save model
            if save_model and "ai_model" in st.session_state:
                try:
                    joblib.dump(st.session_state.ai_model, "student_behaviour_model.joblib")
                    st.success("Model saved to student_behaviour_model.joblib")
                except Exception as e:
                    st.error(f"Could not save model: {e}")
