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

# CENTERED HEADING WITH STYLING
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
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
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
    
    return X_train_balanced, X_test, y_train_balanced, y_test, X.columns, scaler, df

# Load initial data
df = load_data()

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
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", f"{len(df):,}")
    with col2:
        st.metric("Frauds", f"{df['Class'].sum():,}")
    with col3:
        st.metric("Legit", f"{(df['Class']==0).sum():,}")
    with col4:
        fraud_pct = (df['Class'].sum() / len(df)) * 100
        st.metric("Fraud %", f"{fraud_pct:.3f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    # Class Distribution Pie Chart
    with col1:
        st.subheader("Class Distribution")
        class_counts = df['Class'].value_counts()
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
    
    # Amount by Class Histogram
    with col2:
        st.subheader("Amount Distribution by Class")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[df['Class']==0]['Amount'], bins=40, alpha=0.6, label='Legitimate', color='green', edgecolor='black')
        ax.hist(df[df['Class']==1]['Amount'], bins=40, alpha=0.6, label='Fraud', color='red', edgecolor='black')
        ax.set_xlabel('Amount ($)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    # Statistics Table
    with col1:
        st.subheader("Feature Statistics")
        stats_display = df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].head(10)
        st.dataframe(stats_display, use_container_width=True)
    
    # Top Correlated Features
    with col2:
        st.subheader("Top Correlated Features with Fraud")
        corr = df.corr()['Class'].sort_values(ascending=False)[1:11]
        fig, ax = plt.subplots(figsize=(6, 4))
        corr.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Correlation with Fraud')
        ax.set_title('Feature Importance by Correlation')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ============================================
# TAB 2: MODELS
# ============================================
with tab2:
    st.header("🤖 Model Training")
    
    st.info("⚡ Click button to train selected models (2-3 minutes)")
    
    if st.button("🚀 Train Models", key="train_btn"):
        
        # Load preprocessed data
        with st.spinner("📊 Loading and preprocessing data..."):
            X_train, X_test, y_train, y_test, feature_names, scaler, _ = preprocess_data()
            st.write(f"✅ Data loaded: {X_train.shape[0]:,} train samples, {X_test.shape[0]:,} test samples")
        
        st.markdown("---")
        
        models = {}
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(models_to_train)
        
        for idx, model_name in enumerate(models_to_train):
            status_text.text(f"Training {model_name}...")
            
            try:
                if model_name == 'Logistic Regression':
                    model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
                
                elif model_name == 'Random Forest':
                    model = RandomForestClassifier(
                        n_estimators=50, 
                        max_depth=15, 
                        random_state=42, 
                        n_jobs=-1
                    )
                
                elif model_name == 'XGBoost':
                    model = xgb.XGBClassifier(
                        n_estimators=50, 
                        max_depth=7, 
                        learning_rate=0.1,
                        random_state=42, 
                        eval_metric='logloss', 
                        n_jobs=-1,
                        verbosity=0
                    )
                
                elif model_name == 'LightGBM':
                    model = lgb.LGBMClassifier(
                        n_estimators=50, 
                        max_depth=7, 
                        learning_rate=0.1,
                        random_state=42, 
                        verbose=-1, 
                        n_jobs=-1
                    )
                
                elif model_name == 'Gradient Boosting':
                    model = GradientBoostingClassifier(
                        n_estimators=50, 
                        max_depth=5, 
                        learning_rate=0.1,
                        random_state=42
                    )
                
                # Train model
                model.fit(X_train, y_train)
                models[model_name] = model
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                roc_auc = roc_auc_score(y_test, y_proba)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                results[model_name] = {
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'roc_auc': roc_auc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                st.success(f"✅ {model_name}: ROC-AUC = {roc_auc:.4f}, F1 = {f1:.4f}")
            
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / total)
        
        # Cache results in session
        st.session_state.models = models
        st.session_state.results = results
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.X_features = feature_names
        st.session_state.trained = True
        
        st.markdown("---")
        st.balloons()
        st.success("🎉 Training complete! Check the Results tab for detailed analysis.")

# ============================================
# TAB 3: RESULTS
# ============================================
with tab3:
    st.header("📈 Model Results")
    
    if 'trained' not in st.session_state:
        st.warning("⚠️ Train models first in the 'Models' tab!")
    else:
        results = st.session_state.results
        y_test = st.session_state.y_test
        
        if len(results) == 0:
            st.warning("⚠️ No results available yet!")
        else:
            # Model Performance Comparison Table
            st.subheader("📊 Model Performance Comparison")
            
            comparison_data = []
            for name, res in results.items():
                comparison_data.append({
                    'Model': name,
                    'ROC-AUC': f"{res['roc_auc']:.4f}",
                    'Precision': f"{res['precision']:.4f}",
                    'Recall': f"{res['recall']:.4f}",
                    'F1-Score': f"{res['f1']:.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            st.markdown("---")
            
            # ROC Curves Comparison
            st.subheader("📉 ROC Curves")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            
            for (name, res), color in zip(results.items(), colors[:len(results)]):
                fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
                roc_auc = roc_auc_score(y_test, res['y_proba'])
                ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})", linewidth=2.5, color=color)
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            ax.set_title('ROC Curves Comparison', fontsize=13, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Confusion Matrices
            st.subheader("🎯 Confusion Matrices")
            
            if len(results) > 0:
                cols = st.columns(min(len(results), 3))  # Max 3 columns
                col_idx = 0
                
                for name, res in results.items():
                    with cols[col_idx % 3]:
                        cm = confusion_matrix(y_test, res['y_pred'])
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                        ax.set_title(name, fontsize=10, fontweight='bold')
                        ax.set_ylabel('Actual', fontsize=9)
                        ax.set_xlabel('Predicted', fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    col_idx += 1
            else:
                st.info("ℹ️ Confusion matrices will appear after training models.")
            
            st.markdown("---")
            
            # Precision-Recall Curves
            st.subheader("📊 Precision-Recall Curves")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for (name, res), color in zip(results.items(), colors[:len(results)]):
                precision, recall, _ = precision_recall_curve(y_test, res['y_proba'])
                pr_auc = auc(recall, precision)
                ax.plot(recall, precision, label=f"{name} (AUC={pr_auc:.4f})", linewidth=2.5, color=color)
            
            ax.set_xlabel('Recall', fontsize=11)
            ax.set_ylabel('Precision', fontsize=11)
            ax.set_title('Precision-Recall Curves Comparison', fontsize=13, fontweight='bold')
            ax.legend(loc='lower left', fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Classification Reports
            st.subheader("📋 Detailed Classification Report")
            
            selected_model = st.selectbox("Select Model for Details:", list(results.keys()))
            
            if selected_model:
                y_pred = results[selected_model]['y_pred']
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report_dict).transpose()
                
                st.dataframe(report_df.round(4), use_container_width=True)

# ============================================
# TAB 4: BUSINESS METRICS
# ============================================
with tab4:
    st.header("💼 Business Impact Analysis")
    
    if 'trained' not in st.session_state:
        st.warning("⚠️ Train models first in the 'Models' tab!")
    else:
        results = st.session_state.results
        y_test = st.session_state.y_test
        
        if len(results) == 0:
            st.warning("⚠️ No results available yet!")
        else:
            # Model selection
            model_choice = st.selectbox("Select Model for Business Analysis:", list(results.keys()), key="business_model")
            
            y_pred = results[model_choice]['y_pred']
            
            st.subheader("💰 Financial Impact Calculation")
            
            # Input parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_fraud_amount = st.number_input("Average Fraud Amount ($)", value=125, step=10, min_value=1)
            with col2:
                review_cost = st.number_input("Cost to Review False Alarm ($)", value=5, step=1, min_value=1)
            with col3:
                annual_vol = st.number_input("Annual Volume (Millions)", value=10.0, step=1.0, min_value=0.1)
            
            # Calculate metrics
            from sklearn.metrics import confusion_matrix as cm_func
            
            tn, fp, fn, tp = cm_func(y_test, y_pred).ravel()
            
            scale_factor = (annual_vol * 1_000_000) / len(y_test)
            
            fraud_prevented = tp * avg_fraud_amount * scale_factor
            review_costs = fp * review_cost * scale_factor
            fraud_missed = fn * avg_fraud_amount * scale_factor
            net_benefit = fraud_prevented - review_costs - fraud_missed
            
            # Display metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Fraud Prevented (Annual)",
                    f"${fraud_prevented/1_000_000:.2f}M"
                )
            
            with metric_col2:
                st.metric(
                    "Review Costs (Annual)",
                    f"${review_costs/1_000_000:.2f}M"
                )
            
            with metric_col3:
                st.metric(
                    "Fraud Missed (Annual)",
                    f"${fraud_missed/1_000_000:.2f}M"
                )
            
            with metric_col4:
                st.metric(
                    "Net Benefit (Annual)",
                    f"${net_benefit/1_000_000:.2f}M",
                    delta=f"{(net_benefit/max(fraud_prevented, 1)*100):.1f}% ROI"
                )
            
            st.markdown("---")
            
            st.subheader("📊 Prevention Metrics")
            
            detection_rate = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
            false_alarm_rate = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
            approval_rate = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0
            precision_rate = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Fraud Detection Rate", f"{detection_rate:.1f}%")
            
            with col2:
                st.metric("False Alarm Rate", f"{false_alarm_rate:.2f}%")
            
            with col3:
                st.metric("Legitimate Approval Rate", f"{approval_rate:.2f}%")
            
            with col4:
                st.metric("Precision (Correct Alarms)", f"{precision_rate:.1f}%")
            
            st.markdown("---")
            
            st.subheader("📈 Business Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write("**Key Performance Indicators:**")
                st.write(f"• Frauds Detected: {int(tp)} out of {int(tp + fn)} ({detection_rate:.1f}%)")
                st.write(f"• False Positives: {int(fp)} (require manual review)")
                st.write(f"• True Negatives: {int(tn)} (legitimate approved)")
                st.write(f"• Fraud Detection Precision: {precision_rate:.1f}%")
            
            with summary_col2:
                st.write("**Annual Financial Impact:**")
                st.write(f"• Fraud Loss Prevented: ${fraud_prevented:,.0f}")
                st.write(f"• Operations Cost: ${review_costs:,.0f}")
                st.write(f"• Undetected Fraud Loss: ${fraud_missed:,.0f}")
                
                if net_benefit > 0:
                    st.write(f"• **Net Annual Benefit: ${net_benefit:,.0f}** ✅")
                    roi = (net_benefit / 50000) if 50000 > 0 else 0
                    st.write(f"• ROI (vs $50k model cost): {roi:.1f}x")
                else:
                    st.write(f"• **Net Annual Benefit: ${net_benefit:,.0f}** ⚠️")

# ============================================
# FOOTER (CENTERED)
# ============================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:12px;'>"
    "🚨 Advanced Fraud Detection System | Optimized for Speed & Performance | © 2026"
    "</p>",
    unsafe_allow_html=True
)
