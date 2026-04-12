import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, auc, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Fast Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# HEADING
# ============================================
st.markdown(
    """
    <style>
        .centered-title {
            text-align: center;
            color: #e74c3c;
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .centered-subtitle {
            text-align: center;
            color: #95a5a6;
            font-size: 16px;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="centered-title">🚨 Advanced Fraud Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="centered-subtitle">Optimized for Speed & Performance</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# LOAD DATA (CACHED)
# ============================================
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    df = pd.read_csv("https://drive.google.com/uc?id=1YKNYRzROwSMo8DD-TRBD4aukXLnT2JWM")
    return df

@st.cache_data
def preprocess_data(test_size=0.2):
    """Preprocess data once and cache it"""
    df = load_data()

    # Detect fraud label column
    fraud_col = None
    for candidate in ["Class", "Fraud", "Target", "Label"]:
        if candidate in df.columns:
            fraud_col = candidate
            break

    if fraud_col is None:
        st.error("❌ No fraud label column found in dataset. Expected one of: Class, Fraud, Target, Label.")
        return None, None, None, None, None, None, df

    X = df.drop(fraud_col, axis=1)
    y = df[fraud_col]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=42
    )

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, X_test, y_train_balanced, y_test, X.columns, scaler, df, fraud_col

# Load initial data
df = load_data()

# Detect fraud column for metrics
fraud_col = None
for candidate in ["Class", "Fraud", "Target", "Label"]:
    if candidate in df.columns:
        fraud_col = candidate
        break

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("⚙️ Configuration")

models_to_train = st.sidebar.multiselect(
    "Select Models",
    ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting'],
    default=['LightGBM', 'XGBoost']
)

use_cache = st.sidebar.checkbox("Use Cached Results", value=True)

st.sidebar.markdown("---")
st.sidebar.info("✅ Optimized for speed with caching and parallel processing")

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data", 
    "🤖 Models", 
    "📈 Results",
    "💼 Business"
])

# ============================================
# TAB 1: DATA
# ============================================
with tab1:
    st.header("📊 Dataset Overview")

    if fraud_col is None:
        st.error("❌ No fraud label column found in dataset. Please upload the correct Kaggle dataset with 'Class' column.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", f"{len(df):,}")
        with col2:
            st.metric("Frauds", f"{df[fraud_col].sum():,}")
        with col3:
            st.metric("Legit", f"{(df[fraud_col]==0).sum():,}")
        with col4:
            fraud_pct = (df[fraud_col].sum() / len(df)) * 100
            st.metric("Fraud %", f"{fraud_pct:.3f}%")

        st.markdown("---")

        # Class Distribution Pie Chart
        st.subheader("Class Distribution")
        class_counts = df[fraud_col].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax.pie(
            class_counts.values, 
            labels=['Legitimate', 'Fraud'], 
            autopct='%1.1f%%',
            colors=colors, 
            startangle=90
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ============================================
# (Rest of your tabs: Models, Results, Business remain unchanged)
# ============================================
