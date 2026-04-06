import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Job Role Predictor", layout="centered")

# ---------------- TITLE ----------------
st.title("Job Role Predictor")
st.write("Select your skills to find suitable job roles")

# ---------------- DATA ----------------
roles_data = {
    "Data Analyst": ["python", "sql", "excel", "statistics"],
    "Data Scientist": ["python", "machine learning", "statistics", "pandas"],
    "ML Engineer": ["python", "deep learning", "tensorflow"],
    "Frontend Developer": ["html", "css", "javascript", "react"],
    "Backend Developer": ["python", "django", "flask"],
    "Full Stack Developer": ["html", "css", "javascript", "python"],
    "DevOps Engineer": ["docker", "aws", "linux"],
    "Cybersecurity Analyst": ["network", "security", "cryptography"],
    "Software Engineer": ["java", "c++", "algorithms"]
}

df = pd.DataFrame([
    {"Role": role, "Skills": " ".join(skills)}
    for role, skills in roles_data.items()
])

# ---------------- SKILLS ----------------
all_skills = sorted(
    list(set(skill for skills in roles_data.values() for skill in skills)))

selected_skills = st.multiselect("Select your skills", all_skills)

experience = st.slider("Experience (years)", 0, 10, 1)

# ---------------- MODEL ----------------
vectorizer = TfidfVectorizer()
skill_matrix = vectorizer.fit_transform(df["Skills"])

# ---------------- BUTTON ----------------
if st.button("Analyze"):
    if not selected_skills:
        st.warning("Please select at least one skill")
    else:
        user_input = " ".join(selected_skills)
        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, skill_matrix)[0]

        df["Score"] = similarity
        top_roles = df.sort_values(by="Score", ascending=False).head(3)

        # ---------------- RESULTS ----------------
        st.subheader("Top Matches")

        for _, row in top_roles.iterrows():
            role = row["Role"]
            score = int(row["Score"] * 100 + experience * 2)

            required = set(roles_data[role])
            missing = required - set(selected_skills)

            st.write(f"**{role}** — {score}%")

            if missing:
                st.write("Missing skills:", ", ".join(missing))
            else:
                st.write("All skills matched")

            st.write("---")

        # ---------------- STATIC GRAPH ----------------
        st.subheader("Role Comparison")

        sorted_df = df.sort_values(by="Score", ascending=True)

        fig, ax = plt.subplots()
        ax.barh(sorted_df["Role"], sorted_df["Score"])
        ax.set_xlabel("Similarity Score")
        ax.set_ylabel("Role")

        st.pyplot(fig)

        # ---------------- INSIGHT ----------------
        best_role = top_roles.iloc[0]["Role"]
        st.success(f"Best role for you: {best_role}")

# ---------------- FOOTER ----------------
st.write("---")
st.caption("Simple ML-based Job Role Predictor")
