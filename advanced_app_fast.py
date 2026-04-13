import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("🚨 Advanced Fraud Detection System")
st.markdown("Optimized for Speed & Performance")
st.markdown("---")

# ============================================
# LOAD DATA (CACHED)
# ============================================
@st.cache_data
def load_data():
    url = "https://dl.dropboxusercontent.com/scl/fi/rxahy6u2n609wt08vlfz6/creditcard.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.write("Columns:", df.columns.tolist())
st.write("Shape:", df.shape)

# ============================================
# DETECT TARGET COLUMN
# ============================================
fraud_col = None
for col in ["Class", "Fraud", "Target", "Label"]:
    if col in df.columns:
        fraud_col = col
        break

if fraud_col is None:
    st.error("❌ No fraud column found")
    st.stop()

# ============================================
# PREPROCESSING
# ============================================
@st.cache_data
def preprocess(df):
    X = df.drop(fraud_col, axis=1)
    y = df[fraud_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess(df)

# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("⚙️ Settings")

models_selected = st.sidebar.multiselect(
    "Select Models",
    ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"],
    default=["LightGBM", "XGBoost"]
)

train_btn = st.sidebar.button("🚀 Train Models")

# ============================================
# DATA TAB
# ============================================
tab1, tab2 = st.tabs(["📊 Data", "🤖 Models"])

with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(df))
    col2.metric("Frauds", df[fraud_col].sum())
    col3.metric("Legit", (df[fraud_col] == 0).sum())

    st.write(df.head())

# ============================================
# MODEL TRAINING
# ============================================
with tab2:
    st.subheader("Model Training")

    if train_btn:
        results = {}

        for model_name in models_selected:
            st.write(f"Training {model_name}...")

            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)

            elif model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=50)

            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier()

            elif model_name == "XGBoost":
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

            elif model_name == "LightGBM":
                model = lgb.LGBMClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            results[model_name] = acc

            st.write(f"✅ {model_name} Accuracy: {acc:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_title(model_name)
            st.pyplot(fig)

            st.text(classification_report(y_test, y_pred))

        # Final comparison
        st.markdown("### 📊 Model Comparison")
        st.bar_chart(results)

    else:
        st.info("👉 Click 'Train Models' from sidebar")
