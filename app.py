import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Student Behaviour Tracker", layout="wide")
st.title("Student Behaviour Tracker")

# Required columns
required_columns = [
    "Student", "Year Group", "Date", "Behaviour", "Home Life", "Eating Habits",
    "Disabilities", "Interventions", "Safeguarding", "Social", "Comments"
]

# Upload CSV
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
            "Comments": "Jordan has been quiet and disengaged. Referred to mentoring but hasn’t attended yet."
        }
        st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([simulated_entry])], ignore_index=True)

data = st.session_state.data

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Add Entry", "Overview", "Visuals", "Interventions",
    "Comments", "Summary", "Search", "Scoring Guide"
])

# Tab 1: Add Entry
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
        intervention = st.slider("Attending Interventions", 1, 10, 5)
        safeguarding = st.slider("Safeguarding Issues", 1, 10, 5)
        social = st.slider("Social", 1, 10, 5)
        comments = st.text_area("Comments")
        submitted = st.form_submit_button("Submit Entry")

        if submitted:
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
                "Comments": comments
            }
            st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True)
            st.success("Entry added successfully.")

# Tab 2: Overview
with tab2:
    st.header("Student Overview")
    student_list = st.session_state.data["Student"].unique()
    selected_student = st.selectbox("Select Student", student_list)
    student_data = st.session_state.data[st.session_state.data["Student"] == selected_student]

    if not student_data.empty:
        student_data["Total Score"] = student_data[[
            "Behaviour", "Home Life", "Eating Habits", "Disabilities",
            "Interventions", "Safeguarding", "Social"
        ]].sum(axis=1)

        latest = student_data.sort_values("Date", ascending=False).iloc[0]
        avg_score = student_data["Total Score"].mean()

        if avg_score < 25:
            status = "High Risk"
        elif avg_score < 40:
            status = "Monitor"
        else:
            status = "Stable"

        st.metric("Latest Behaviour Score", latest["Total Score"])
        st.metric("Average Behaviour Score", round(avg_score, 1))
        st.metric("Risk Category", status)
        st.write("Latest Entry:")
        st.dataframe(latest)

# Tab 3: Visuals
with tab3:
    st.header("Behaviour Trends")

    if not st.session_state.data.empty:
        students = st.multiselect("Select Students", st.session_state.data["Student"].unique())

        if students:
            filtered = st.session_state.data[st.session_state.data["Student"].isin(students)].copy()
            filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")
            filtered["Total Score"] = filtered[[
                "Behaviour", "Home Life", "Eating Habits", "Disabilities",
                "Interventions", "Safeguarding", "Social"
            ]].sum(axis=1)

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

# Tab 4: Interventions
with tab4:
    st.header("Intervention Log")
    st.dataframe(st.session_state.data[["Student", "Year Group", "Date", "Interventions", "Comments"]])

# Tab 5: Comments
with tab5:
    st.header("Comments and Notes")
    st.dataframe(st.session_state.data[["Student", "Year Group", "Date", "Comments"]])

# Tab 6: Summary
with tab6:
    st.header("Student Summary")
    student = st.selectbox("Select Student for Summary", st.session_state.data["Student"].unique())
    st.dataframe(st.session_state.data[st.session_state.data["Student"] == student])

# Tab 7: Search
with tab7:
    st.header("Search Entries")
    query = st.text_input("Search by student name, year group, or keyword")
    if query:
        results = st.session_state.data[st.session_state.data.apply(lambda row: query.lower() in str(row).lower(), axis=1)]
        st.dataframe(results)

# Tab 8: Scoring Guide and Downloads
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

    st.subheader("Download Behaviour Data")

    all_csv = st.session_state.data.to_csv(index=False).encode("utf-8")
    st.download_button("Download All Student Data", all_csv, "all_student_data.csv", "text/csv")

    selected_student = st.selectbox("Student for Individual Export", st.session_state.data["Student"].unique(), key="export_student")
    student_data = st.session_state.data[st.session_state.data["Student"] == selected_student]
    if not student_data.empty:
        student_csv = student_data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Student Data", student_csv, f"{selected_student}_student_data.csv", "text/csv")                