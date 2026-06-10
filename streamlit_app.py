import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

st.set_page_config(
    page_title="NYC Restaurant Inspection Dashboard 🍽️",
    layout="centered",
    page_icon="🍽️",
)

st.sidebar.title("NYC Restaurant Inspection 🍽️")
page = st.sidebar.selectbox("Select Page", ["Introduction 📘", "Visualization 📊", "Prediction 🔮"])

df = pd.read_csv("cleaned_restaurant_data.csv.gz")

st.image("NYC Restaurant Photo.png")


## Introduction Page
if page == "Introduction 📘":
    st.subheader("01 Introduction 📘")

    st.title("🍽️ NYC Restaurant Inspection Score Prediction")

    st.markdown("""
    Many customers choose restaurants based on **price, location, cuisine type, or online reviews**.
    However, these factors do not always show the real **food safety risk** of a restaurant.
    """)

    st.info(
        "Our project uses NYC restaurant inspection data to analyze food safety patterns "
        "and predict possible inspection grades using linear regression."
    )

    st.markdown("### 🔍 What We Focus On")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 📍 Location")
        st.caption("Borough and ZIP code may reflect different business environments.")

    with col2:
        st.markdown("#### 🍜 Cuisine")
        st.caption("Different foods may need different storage and cooking methods.")

    with col3:
        st.markdown("#### ⚠️ Critical Flag")
        st.caption("Shows whether the problem is a serious food safety issue.")

    with col4:
        st.markdown("#### 🔎 Inspection Type")
        st.caption("Initial inspections and re-inspections may represent different situations.")

    st.markdown("### 📊 Project Roadmap")
    st.markdown("""
    **Visualization** → explore inspection score patterns

    **Linear Regression** → predict possible inspection grades

    **Model Evaluation** → compare actual grades with predicted grades
    """)

    st.divider()

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    st.markdown("##### Missing Values")
    missing = df.isnull().sum()
    st.write(missing)
    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ You have missing values")

    st.markdown("##### Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())


## Visualization Page
elif page == "Visualization 📊":
    st.subheader("02 Data Visualization 📊")
    st.markdown("> ⚠️ **Reminder:** Higher score = more violations = worse food safety.")

    st.markdown("##### Distribution of Restaurant Inspection Scores")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["SCORE"], bins=30, ax=ax1)
    ax1.set_title("Distribution of Restaurant Inspection Scores")
    ax1.set_xlabel("Inspection Score")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    st.markdown("##### Average Inspection Score by Borough")
    boro_score = df.groupby("BORO")["SCORE"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(x=boro_score.index, y=boro_score.values, ax=ax2)
    ax2.set_title("Average Inspection Score by Borough")
    ax2.set_xlabel("Borough")
    ax2.set_ylabel("Average Score")
    plt.xticks(rotation=30)
    st.pyplot(fig2)

    st.markdown("##### Restaurant Grade Distribution")
    grade_counts = df[df["GRADE"].isin(["A", "B", "C"])]["GRADE"].value_counts()
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.barplot(x=grade_counts.index, y=grade_counts.values, ax=ax3)
    ax3.set_title("Restaurant Grade Distribution")
    ax3.set_xlabel("Grade")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)

    st.markdown("##### Top 10 Cuisine Types by Average Inspection Score")
    cuisine_score = (
        df.groupby("CUISINE DESCRIPTION")["SCORE"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=cuisine_score.values, y=cuisine_score.index, ax=ax4)
    ax4.set_title("Top 10 Cuisine Types by Average Inspection Score")
    ax4.set_xlabel("Average Score")
    ax4.set_ylabel("Cuisine Type")
    st.pyplot(fig4)

    st.markdown("##### Top 10 Most Common Restaurant Violations")
    violation_counts = (
        df["VIOLATION DESCRIPTION"]
        .dropna()
        .value_counts()
        .head(10)
    )
    fig5, ax5 = plt.subplots(figsize=(15, 7))
    sns.barplot(x=violation_counts.values, y=violation_counts.index, ax=ax5)
    ax5.set_title("Top 10 Most Common Restaurant Violations")
    ax5.set_xlabel("Number of Violations")
    ax5.set_ylabel("Violation Description")
    plt.tight_layout()
    st.pyplot(fig5)


## Prediction Page
elif page == "Prediction 🔮":
    st.subheader("03 Prediction with Linear Regression 🔮")

    st.markdown("> ⚠️ **Note:** In NYC restaurant inspections, a **higher score = more violations**. A score of 0–13 → Grade A, 14–27 → Grade B, 28+ → Grade C.")

    ## Data Preprocessing
    df2 = df.copy()
    df2 = df2.dropna()

    features = ["BORO_ENC", "CUISINE DESCRIPTION_ENC", "CRITICAL FLAG_ENC", "INSPECTION TYPE_ENC", "ZIPCODE"]
    target = "SCORE"

    ## i) X and y
    X = df2[features]
    y = df2[target]

    rows = st.slider("Select number of rows to display", 5, 20, 5)
    st.dataframe(X.head(rows))
    st.dataframe(y.head(rows))

    st.markdown("##### Encoding Key")
    st.write("BORO_ENC: 0 = Manhattan, 1 = Queens, 2 = Staten Island, 3 = Brooklyn, 4 = Bronx")
    st.write("CUISINE DESCRIPTION_ENC: 0 = Afghan, 1 = Chinese, 2 = American, etc.")
    st.write("CRITICAL FLAG_ENC: 0 = Critical, 1 = Not Critical, 2 = Not Applicable")
    st.write("INSPECTION TYPE_ENC: 0 = Cycle Inspection / Initial Inspection, 1 = Pre-permit / Initial Inspection, etc.")
    st.write("SCORE: inspection score (target variable) — higher score = more violations")

    ## ii) train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ## iii) Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    ## iv) Prediction
    predictions = model.predict(X_test)

    ## v) Evaluation
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)

    st.write(f"- **MAE**: {mae:,.2f}")
    st.write(f"- **MSE**: {mse:,.2f}")
    st.write(f"- **R² Score**: {r2:,.3f}")

    st.success(f"Model MAE: {np.round(mae, 2)} score points on average")

    ## vi) Actual vs Predicted plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", linewidth=2)
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Actual vs Predicted Inspection Score")
    st.pyplot(fig)

    st.divider()

    ## vii) Interactive Prediction
    st.markdown("#### Predict a Restaurant's Inspection Score")

    BORO_ENC = {"MANHATTAN": 0, "QUEENS": 1, "STATEN ISLAND": 2, "BROOKLYN": 3, "BRONX": 4}
    CRITICAL_FLAG_ENC = {"Critical": 0, "Not Critical": 1, "Not Applicable": 2}
    INSPECTION_TYPE_ENC = {
        "Cycle Inspection / Initial Inspection": 0,
        "Pre-permit (Operational) / Initial Inspection": 1,
        "Pre-permit (Operational) / Compliance Inspection": 2,
        "Pre-permit (Operational) / Re-inspection": 3,
        "Cycle Inspection / Re-inspection": 4,
        "Cycle Inspection / Compliance Inspection": 5,
        "Cycle Inspection / Reopening Inspection": 6,
        "Inter-Agency Task Force / Initial Inspection": 7,
        "Pre-permit (Non-operational) / Re-inspection": 8,
        "Pre-permit (Non-operational) / Initial Inspection": 9,
        "Pre-permit (Operational) / Reopening Inspection": 10,
        "Pre-permit (Operational) / Second Compliance Inspection": 11,
        "Administrative Miscellaneous / Re-inspection": 12,
        "Cycle Inspection / Second Compliance Inspection": 13,
        "Pre-permit (Non-operational) / Compliance Inspection": 14,
        "Trans Fat / Re-inspection": 15,
        "Inter-Agency Task Force / Re-inspection": 16,
        "Smoke-Free Air Act / Re-inspection": 17,
        "Calorie Posting / Re-inspection": 18,
        "Smoke-Free Air Act / Initial Inspection": 19,
    }
    CUISINE_ENC = (
        df.drop_duplicates("CUISINE DESCRIPTION")
        .set_index("CUISINE DESCRIPTION")["CUISINE DESCRIPTION_ENC"]
        .to_dict()
    )

    boro_input = st.selectbox("Select Borough", list(BORO_ENC.keys()))
    cuisine_input = st.selectbox("Select Cuisine Type", sorted(CUISINE_ENC.keys()))
    critical_input = st.selectbox("Select Critical Flag", list(CRITICAL_FLAG_ENC.keys()))
    inspection_input = st.selectbox("Select Inspection Type", list(INSPECTION_TYPE_ENC.keys()))
    zipcode_input = st.number_input("Enter ZIP Code", min_value=10001, max_value=11697, value=10001, step=1)

    if st.button("Predict Score"):
        input_data = pd.DataFrame({
            "BORO_ENC": [BORO_ENC[boro_input]],
            "CUISINE DESCRIPTION_ENC": [CUISINE_ENC[cuisine_input]],
            "CRITICAL FLAG_ENC": [CRITICAL_FLAG_ENC[critical_input]],
            "INSPECTION TYPE_ENC": [INSPECTION_TYPE_ENC[inspection_input]],
            "ZIPCODE": [zipcode_input],
        })

        predicted_score = max(0, round(model.predict(input_data)[0], 1))

        if predicted_score <= 13:
            grade = "A"
        elif predicted_score <= 27:
            grade = "B"
        else:
            grade = "C"

        st.success(f"Predicted Inspection Score: **{predicted_score}** → Grade **{grade}**")
        st.write(f"A {cuisine_input} restaurant in {boro_input} with a {critical_input} violation flag is estimated to score **{predicted_score}**.")
