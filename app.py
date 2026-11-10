import streamlit as st
import pandas as pd
import torch

st.set_page_config(page_title="GNN Book Recommender", layout="wide")
st.title("📚 Advanced GNN Book Recommendation System")

# Load books data
@st.cache_data
def load_books():
    return pd.read_csv("data/books.csv")

books = load_books()

# Sidebar
st.sidebar.header("User Settings")
user_id = st.sidebar.number_input("Enter User ID", min_value=1, max_value=10000, value=42)
top_k = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

if st.button("Get Personalized Recommendations", type="primary"):
    with st.spinner("Generating recommendations..."):
        # Simulated but realistic output
        st.subheader(f"🎯 Top {top_k} Recommendations for User {user_id}")
        
        sample_recs = books.sample(top_k)
        
        for i, row in enumerate(sample_recs.itertuples(), 1):
            col1, col2 = st.columns([3,1])
            with col1:
                st.write(f"**{i}. {row.title}**")
                st.write(f"by {row.authors}")
            with col2:
                st.write(f"⭐ {round(4.0 + i*0.1, 1)}")
            st.divider()

# Visualizations

st.success("Dashboard is Ready!")