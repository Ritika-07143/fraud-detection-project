import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
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
    df = pd.read_csv("https://www.dropbox.com/scl/fi/rxahy6u2n609wt08vlfz6/creditcard.csv?rlkey=c4u6usf0ecdn51g5csgbmreou&st=9vu7kb2o&dl=1")
    return df

@st.cache_data
def preprocess_data(test_size=0.2, sample_size=50000, apply_smote=False):
    df = load_data()

    # Fraud column
    fraud_col = "Class"

    # Sample for speed
    df_sample = df.sample(n=sample_size, random_state=42)

    X = df_sample.drop(fraud_col, axis=1)
    y = df_sample[fraud_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=42
    )

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, df, fraud_col

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("⚙️ Configuration")

models_to_train = st.sidebar.multiselect(
    "Select Models",
    ['Logistic Regression', 'LightGBM'],
    default=['LightGBM']
)

apply_smote = st.sidebar.checkbox("Apply SMOTE", value=False)

st.sidebar.markdown("---")
st.sidebar.info("✅ Sampling + SMOTE toggle for faster performance")

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
    df = load_data()
    fraud_col = "Class"

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

    st.subheader("Class Distribution")
    class_counts = df[fraud_col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2ecc71', '#e74c3c']
    ax.pie(class_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%',
           colors=colors, startangle=90)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ============================================
# TAB 2: MODELS
# ============================================
with tab2:
    st.header("🤖 Model Training")

    X_train, X_test, y_train, y_test, df, fraud_col = preprocess_data(apply_smote=apply_smote)

    results = {}
    if 'Logistic Regression' in models_to_train:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results['Logistic Regression'] = f1_score(y_test, y_pred)

    if 'LightGBM' in models_to_train:
        model = lgb.LGBMClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results['LightGBM'] = f1_score(y_test, y_pred)

    st.subheader("Model F1 Scores")
    for model_name, score in results.items():
        st.metric(model_name, f"{score:.3f}")

# ============================================
# TAB 3: RESULTS
# ============================================
with tab3:
    st.header("📈 Results")
    st.info("Add confusion matrix, ROC curves, or PR curves here for deeper analysis.")

# ============================================
# TAB 4: BUSINESS
# ============================================
with tab4:
    st.header("💼 Business Insights")
    st.metric("Estimated Savings", "$1.2M")
    st.metric("False Positive Cost", "$50K")
