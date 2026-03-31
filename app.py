import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score
)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Shield | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0a0e1a; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1530 50%, #0a1628 100%); }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.hero-banner {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
    border: 1px solid #00d4ff33; border-radius: 16px;
    padding: 2.5rem 2rem; margin-bottom: 2rem;
    text-align: center; box-shadow: 0 0 40px #00d4ff22;
}
.hero-banner h1 { color: #00d4ff; font-size: 2.4rem; letter-spacing: 2px; margin: 0; text-shadow: 0 0 20px #00d4ff88; }
.hero-banner p  { color: #a0b4c8; font-size: 1rem; margin-top: 0.5rem; }
.metric-card {
    background: linear-gradient(145deg, #111827, #1a2332);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 1.2rem 1.5rem; text-align: center;
    box-shadow: 0 4px 20px rgba(0,212,255,0.08); transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card .label { color: #6b8cae; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px; }
.metric-card .value { color: #00d4ff; font-size: 2rem; font-weight: 700; font-family: 'Space Mono', monospace; }
.smote-card {
    background: linear-gradient(145deg, #0d1f0d, #1a2f1a);
    border: 1px solid #00cc6644; border-radius: 12px;
    padding: 1.2rem 1.5rem; text-align: center;
    box-shadow: 0 4px 20px rgba(0,204,102,0.10);
}
.smote-card .label { color: #5a9e6b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px; }
.smote-card .value { color: #00ff88; font-size: 2rem; font-weight: 700; font-family: 'Space Mono', monospace; }
.info-box {
    background: linear-gradient(145deg, #0d1a2e, #0f2040);
    border-left: 4px solid #00d4ff; border-radius: 8px;
    padding: 1rem 1.2rem; margin: 1rem 0;
    color: #a0b4c8; font-size: 0.9rem;
}
.info-box strong { color: #00d4ff; }
.fraud-alert {
    background: linear-gradient(135deg, #3d0000, #5a0000);
    border: 2px solid #ff4444; border-radius: 12px; padding: 1.5rem;
    text-align: center; box-shadow: 0 0 30px #ff444444; animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{box-shadow:0 0 20px #ff444455;} 50%{box-shadow:0 0 40px #ff4444aa;} }
.legit-badge {
    background: linear-gradient(135deg, #003d1a, #005522);
    border: 2px solid #00cc66; border-radius: 12px; padding: 1.5rem;
    text-align: center; box-shadow: 0 0 30px #00cc6633;
}
.fraud-alert h2, .legit-badge h2 { font-family: 'Space Mono', monospace; font-size: 1.5rem; margin: 0; }
.fraud-alert h2 { color: #ff6666; }
.legit-badge h2 { color: #00ff88; }
.section-title {
    color: #00d4ff; font-family: 'Space Mono', monospace;
    font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f; padding-bottom: 0.5rem; margin-bottom: 1rem;
}
.compare-better { color: #00ff88; font-weight: 700; }
.compare-worse  { color: #ff6666; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset(n_samples=10000):
    np.random.seed(42)
    n_fraud = int(n_samples * 0.02)
    n_legit = n_samples - n_fraud
    legit = pd.DataFrame({
        'Amount': np.random.exponential(scale=80, size=n_legit),
        'Hour':   np.random.randint(0, 24, n_legit),
        'V1': np.random.normal(0, 1, n_legit), 'V2': np.random.normal(0, 1, n_legit),
        'V3': np.random.normal(0, 1, n_legit), 'V4': np.random.normal(0, 1, n_legit),
        'V5': np.random.normal(0, 1, n_legit), 'V6': np.random.normal(0, 1, n_legit),
        'V7': np.random.normal(0, 1, n_legit), 'V8': np.random.normal(0, 1, n_legit),
        'Class': 0
    })
    fraud = pd.DataFrame({
        'Amount': np.random.exponential(scale=200, size=n_fraud),
        'Hour':   np.random.choice([0,1,2,3,22,23], n_fraud),
        'V1': np.random.normal(-3, 1.5, n_fraud), 'V2': np.random.normal(2, 1.5, n_fraud),
        'V3': np.random.normal(-2.5, 1, n_fraud), 'V4': np.random.normal(2.5, 1, n_fraud),
        'V5': np.random.normal(-1.5, 1, n_fraud), 'V6': np.random.normal(1.5, 1, n_fraud),
        'V7': np.random.normal(-2, 1, n_fraud),   'V8': np.random.normal(2, 1, n_fraud),
        'Class': 1
    })
    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df['Amount'] = df['Amount'].clip(lower=0.5, upper=5000)
    return df


FEATURES = ['Amount', 'Hour', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
SMOTE_VARIANTS = {
    "SMOTE (Standard)": SMOTE(random_state=42),
    "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
    "SVM-SMOTE":        SVMSMOTE(random_state=42),
}


def _fit_evaluate(X_train, y_train, X_test, y_test, model_type):
    if model_type == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    else:
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return clf, {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "confusion": confusion_matrix(y_test, y_pred),
        "report":    classification_report(y_test, y_pred, output_dict=True),
        "y_test":    y_test,
        "y_prob":    y_prob,
        "feature_names": FEATURES,
        "feature_importance": clf.feature_importances_ if model_type == "Random Forest" else None,
    }


@st.cache_resource
def train_all(model_type, smote_variant):
    df = generate_dataset()
    X = df[FEATURES].values
    y = df['Class'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    clf_no, metrics_no = _fit_evaluate(X_train, y_train, X_test, y_test, model_type)

    sm = SMOTE_VARIANTS[smote_variant]
    X_res, y_res = sm.fit_resample(X_train, y_train)
    clf_sm, metrics_sm = _fit_evaluate(X_res, y_res, X_test, y_test, model_type)

    before = dict(zip(*np.unique(y_train, return_counts=True)))
    after  = dict(zip(*np.unique(y_res,   return_counts=True)))
    return scaler, clf_sm, metrics_no, metrics_sm, before, after, X_train, X_res, y_res


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fraud Shield")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "🔬 SMOTE Analysis",
        "⚖️ SMOTE vs No-SMOTE",
        "🔍 Predict Transaction",
        "📊 Model Insights",
        "📂 Dataset Explorer",
    ])
    st.markdown("---")
    model_choice = st.selectbox("Classifier", ["Random Forest", "Logistic Regression"])
    smote_choice = st.selectbox("SMOTE Variant", list(SMOTE_VARIANTS.keys()))
    st.markdown("---")
    st.markdown("<small style='color:#6b8cae'>Built with Streamlit · sklearn · imbalanced-learn · Plotly</small>", unsafe_allow_html=True)

df = generate_dataset()
scaler, model, metrics_no, metrics_sm, before_counts, after_counts, X_train_orig, X_res, y_res = train_all(model_choice, smote_choice)
metrics = metrics_sm


# ═══════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <h1>🛡️ FRAUD SHIELD</h1>
        <p>Credit Card Fraud Detection · SMOTE-Balanced ML Pipeline</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in zip(
        [c1,c2,c3,c4],
        ["Accuracy","ROC-AUC","F1 Score","Fraud Rate"],
        [f"{metrics['accuracy']:.1%}", f"{metrics['roc_auc']:.3f}",
         f"{metrics['f1']:.3f}", f"{df['Class'].mean()*100:.1f}%"]
    ):
        col.markdown(f'<div class="metric-card"><div class="label">{lbl}</div><div class="value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Transaction Amount Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Amount', color='Class',
                           color_discrete_map={0:'#00d4ff',1:'#ff4444'},
                           barmode='overlay', nbins=60, opacity=0.75, template='plotly_dark')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<p class="section-title">Transactions by Hour</p>', unsafe_allow_html=True)
        hourly = df.groupby(['Hour','Class']).size().reset_index(name='Count')
        fig2 = px.line(hourly, x='Hour', y='Count', color='Class',
                       color_discrete_map={0:'#00d4ff',1:'#ff4444'}, markers=True, template='plotly_dark')
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<p class="section-title">Class Distribution (Original)</p>', unsafe_allow_html=True)
        counts = df['Class'].value_counts().reset_index()
        counts.columns = ['Class','Count']
        counts['Label'] = counts['Class'].map({0:'Legitimate',1:'Fraudulent'})
        fig3 = px.pie(counts, values='Count', names='Label',
                      color='Label', color_discrete_map={'Legitimate':'#00d4ff','Fraudulent':'#ff4444'},
                      hole=0.55, template='plotly_dark')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.markdown('<p class="section-title">ROC Curve (SMOTE vs No-SMOTE)</p>', unsafe_allow_html=True)
        fpr,  tpr,  _ = roc_curve(metrics['y_test'],    metrics['y_prob'])
        fpr2, tpr2, _ = roc_curve(metrics_no['y_test'], metrics_no['y_prob'])
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=fpr,  y=tpr,  mode='lines', name=f'SMOTE   AUC={metrics["roc_auc"]:.3f}',    line=dict(color='#00ff88', width=2)))
        fig4.add_trace(go.Scatter(x=fpr2, y=tpr2, mode='lines', name=f'No-SMOTE AUC={metrics_no["roc_auc"]:.3f}', line=dict(color='#00d4ff', width=2, dash='dot')))
        fig4.add_trace(go.Scatter(x=[0,1],y=[0,1], mode='lines', name='Random', line=dict(color='#444', dash='dash')))
        fig4.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10),
                           xaxis_title='FPR', yaxis_title='TPR')
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════
# SMOTE ANALYSIS
# ═══════════════════════════════════════════════
elif page == "🔬 SMOTE Analysis":
    st.markdown('<h2 style="color:#00ff88;font-family:Space Mono">🔬 SMOTE Analysis</h2>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <strong>What is SMOTE?</strong> Synthetic Minority Over-sampling Technique (SMOTE) generates
        <em>synthetic</em> fraud samples by interpolating between existing minority-class instances in
        feature space — it doesn't duplicate rows, it creates brand-new plausible examples.<br><br>
        <strong>Active variant:</strong> {smote_choice}
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="label">Legit (before)</div><div class="value">{before_counts.get(0,0):,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="label">Fraud (before)</div><div class="value">{before_counts.get(1,0):,}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="smote-card"><div class="label">Legit (after)</div><div class="value">{after_counts.get(0,0):,}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="smote-card"><div class="label">Fraud (after)</div><div class="value">{after_counts.get(1,0):,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Before vs After — Bar Chart</p>', unsafe_allow_html=True)
        bar_df = pd.DataFrame({
            'Stage': ['Before','Before','After','After'],
            'Class': ['Legitimate','Fraudulent','Legitimate','Fraudulent'],
            'Count': [before_counts.get(0,0), before_counts.get(1,0),
                      after_counts.get(0,0),  after_counts.get(1,0)],
        })
        fig = px.bar(bar_df, x='Stage', y='Count', color='Class',
                     color_discrete_map={'Legitimate':'#00d4ff','Fraudulent':'#ff4444'},
                     barmode='group', template='plotly_dark', text='Count')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Fraud Class Ratio</p>', unsafe_allow_html=True)
        rb = before_counts.get(1,0) / sum(before_counts.values()) * 100
        ra = after_counts.get(1,0)  / sum(after_counts.values())  * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Before SMOTE', x=['Fraud %'], y=[rb],
                              marker_color='#ff4444', text=[f'{rb:.1f}%'], textposition='outside'))
        fig2.add_trace(go.Bar(name='After SMOTE',  x=['Fraud %'], y=[ra],
                              marker_color='#00ff88', text=[f'{ra:.1f}%'], textposition='outside'))
        fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[0, ra*1.6]),
                           margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # PCA scatter
    st.markdown('<p class="section-title">PCA 2D View — Original vs Synthetic Samples</p>', unsafe_allow_html=True)
    st.caption("🟢 Synthetic fraud (SMOTE-generated)  🔴 Original fraud  🔵 Legitimate")

    n_orig_train = len(X_train_orig)
    n_synth      = len(X_res) - n_orig_train
    y_train_orig_labels = np.where(
        np.concatenate([
            np.zeros(int((1-0.02)*8000)),
            np.ones(int(0.02*8000))
        ])[:n_orig_train] == 0, 0, 1
    )
    # simpler: label by y_res for original part
    label_arr = []
    for i, cls in enumerate(y_res):
        if i < n_orig_train:
            label_arr.append('Legitimate' if cls == 0 else 'Original Fraud')
        else:
            label_arr.append('Synthetic Fraud')

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_res)
    pca_df = pd.DataFrame({'PC1': coords[:,0], 'PC2': coords[:,1], 'Type': label_arr})
    color_map = {'Legitimate':'#00d4ff88','Original Fraud':'#ff4444','Synthetic Fraud':'#00ff88'}
    fig3 = px.scatter(pca_df, x='PC1', y='PC2', color='Type', color_discrete_map=color_map,
                      opacity=0.5, template='plotly_dark',
                      category_orders={'Type':['Legitimate','Original Fraud','Synthetic Fraud']})
    fig3.update_traces(marker=dict(size=3))
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # Variant cards
    st.markdown('<p class="section-title">SMOTE Variant Guide</p>', unsafe_allow_html=True)
    descs = [
        ("SMOTE (Standard)", "#00d4ff",
         "Interpolates between a minority sample and one of its k nearest neighbours. Works well on most imbalanced datasets."),
        ("Borderline-SMOTE", "#ffaa00",
         "Over-samples only borderline minority instances near the decision boundary — where the classifier struggles most."),
        ("SVM-SMOTE", "#aa00ff",
         "Uses an SVM to identify support vectors in the minority class and generates samples along the SVM margin for complex boundaries."),
    ]
    cols = st.columns(3)
    for col, (name, color, desc) in zip(cols, descs):
        active_border = f"2px solid {color}" if name == smote_choice else "1px solid #1e3a5f"
        col.markdown(f"""
        <div style="background:#111827;border:{active_border};border-radius:10px;padding:1rem;min-height:150px">
            <div style="color:{color};font-family:Space Mono;font-size:0.8rem;font-weight:700">{name}</div>
            <div style="color:#a0b4c8;font-size:0.82rem;margin-top:0.5rem">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SMOTE vs NO-SMOTE
# ═══════════════════════════════════════════════
elif page == "⚖️ SMOTE vs No-SMOTE":
    st.markdown('<h2 style="color:#00d4ff;font-family:Space Mono">⚖️ SMOTE vs No-SMOTE Comparison</h2>', unsafe_allow_html=True)

    metric_pairs = [
        ("F1 Score",       metrics_no['f1'],                    metrics_sm['f1']),
        ("ROC-AUC",        metrics_no['roc_auc'],               metrics_sm['roc_auc']),
        ("Fraud Recall",   metrics_no['report']['1']['recall'], metrics_sm['report']['1']['recall']),
    ]
    cols = st.columns(3)
    for col, (lbl, no_val, sm_val) in zip(cols, metric_pairs):
        better = sm_val >= no_val
        col.markdown(f"""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:1rem;text-align:center">
            <div style="color:#6b8cae;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">{lbl}</div>
            <div style="display:flex;justify-content:space-around;margin-top:0.8rem">
                <div>
                    <div style="color:#888;font-size:0.7rem">No SMOTE</div>
                    <div style="color:#00d4ff;font-size:1.4rem;font-family:Space Mono">{no_val:.3f}</div>
                </div>
                <div style="color:#555;font-size:1.4rem;padding-top:0.3rem">→</div>
                <div>
                    <div style="color:#888;font-size:0.7rem">With SMOTE</div>
                    <div style="color:{'#00ff88' if better else '#ff6666'};font-size:1.4rem;font-family:Space Mono">{sm_val:.3f}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">All Metrics — Grouped Bar</p>', unsafe_allow_html=True)
        mdf = pd.DataFrame({
            'Metric':     ['Accuracy','F1','ROC-AUC','Fraud Recall','Fraud Precision'],
            'No SMOTE':   [metrics_no['accuracy'], metrics_no['f1'], metrics_no['roc_auc'],
                           metrics_no['report']['1']['recall'], metrics_no['report']['1']['precision']],
            'With SMOTE': [metrics_sm['accuracy'], metrics_sm['f1'], metrics_sm['roc_auc'],
                           metrics_sm['report']['1']['recall'], metrics_sm['report']['1']['precision']],
        }).melt(id_vars='Metric', var_name='Method', value_name='Score')
        fig = px.bar(mdf, x='Metric', y='Score', color='Method',
                     color_discrete_map={'No SMOTE':'#00d4ff','With SMOTE':'#00ff88'},
                     barmode='group', template='plotly_dark', text_auto='.3f')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          yaxis=dict(range=[0,1.05]), margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">ROC Curve Overlay</p>', unsafe_allow_html=True)
        fpr_no, tpr_no, _ = roc_curve(metrics_no['y_test'], metrics_no['y_prob'])
        fpr_sm, tpr_sm, _ = roc_curve(metrics_sm['y_test'], metrics_sm['y_prob'])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fpr_no, y=tpr_no, mode='lines',
                                  name=f'No SMOTE  AUC={metrics_no["roc_auc"]:.3f}',
                                  line=dict(color='#00d4ff', width=2, dash='dot')))
        fig2.add_trace(go.Scatter(x=fpr_sm, y=tpr_sm, mode='lines',
                                  name=f'With SMOTE AUC={metrics_sm["roc_auc"]:.3f}',
                                  line=dict(color='#00ff88', width=2)))
        fig2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                                  line=dict(color='#444', dash='dash')))
        fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)', xaxis_title='FPR', yaxis_title='TPR')
        st.plotly_chart(fig2, use_container_width=True)

    # Confusion matrices side by side
    st.markdown('<p class="section-title">Confusion Matrices</p>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    for col, cm, title in [(col3, metrics_no['confusion'], "No SMOTE"), (col4, metrics_sm['confusion'], "With SMOTE")]:
        with col:
            st.caption(title)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicted", y="Actual"),
                               x=['Legit','Fraud'], y=['Legit','Fraud'], template='plotly_dark')
            fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════
elif page == "🔍 Predict Transaction":
    st.markdown('<h2 style="color:#00d4ff;font-family:Space Mono">🔍 Predict Transaction</h2>', unsafe_allow_html=True)
    st.markdown(f"Model: **{model_choice}** trained with **{smote_choice}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount ($)", min_value=0.01, max_value=5000.0, value=150.0, step=0.01)
        hour   = st.slider("Transaction Hour (0-23)", 0, 23, 14)
        v1     = st.slider("V1", -5.0, 5.0, 0.0, 0.01)
    with col2:
        v2 = st.slider("V2", -5.0, 5.0, 0.0, 0.01)
        v3 = st.slider("V3", -5.0, 5.0, 0.0, 0.01)
        v4 = st.slider("V4", -5.0, 5.0, 0.0, 0.01)
    with col3:
        v5 = st.slider("V5", -5.0, 5.0, 0.0, 0.01)
        v6 = st.slider("V6", -5.0, 5.0, 0.0, 0.01)
        v7 = st.slider("V7", -5.0, 5.0, 0.0, 0.01)
        v8 = st.slider("V8", -5.0, 5.0, 0.0, 0.01)

    if st.button("🔎 Analyze Transaction", use_container_width=True):
        inp  = scaler.transform(np.array([[amount, hour, v1, v2, v3, v4, v5, v6, v7, v8]]))
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0]
        st.markdown("---")
        if pred == 1:
            st.markdown(f'<div class="fraud-alert"><h2>⚠️ FRAUDULENT TRANSACTION DETECTED</h2><p style="color:#ffaaaa;margin-top:0.5rem">Fraud Probability: <strong>{prob[1]:.1%}</strong></p><p style="color:#ff8888;font-size:0.85rem">Recommend: Block & alert cardholder immediately.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="legit-badge"><h2>✅ LEGITIMATE TRANSACTION</h2><p style="color:#aaffcc;margin-top:0.5rem">Legit Probability: <strong>{prob[0]:.1%}</strong></p><p style="color:#88ffbb;font-size:0.85rem">No anomalies detected.</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=['Legitimate','Fraudulent'], y=[prob[0], prob[1]],
                               marker_color=['#00d4ff','#ff4444'],
                               text=[f'{p:.1%}' for p in prob], textposition='outside'))
        fig.update_layout(title='Prediction Confidence', template='plotly_dark',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          yaxis=dict(range=[0,1.15]))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# MODEL INSIGHTS
# ═══════════════════════════════════════════════
elif page == "📊 Model Insights":
    st.markdown('<h2 style="color:#00d4ff;font-family:Space Mono">📊 Model Insights</h2>', unsafe_allow_html=True)
    st.caption(f"**{model_choice}** · **{smote_choice}**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
        fig = px.imshow(metrics['confusion'], text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Legit','Fraud'], y=['Legit','Fraud'], template='plotly_dark')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if model_choice == "Random Forest":
            st.markdown('<p class="section-title">Feature Importances</p>', unsafe_allow_html=True)
            fi_df = pd.DataFrame({'Feature': FEATURES, 'Importance': metrics['feature_importance']}).sort_values('Importance', ascending=True)
            fig2 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                          color='Importance', color_continuous_scale='Blues', template='plotly_dark')
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown('<p class="section-title">Precision-Recall Curve</p>', unsafe_allow_html=True)
            prec, rec, _ = precision_recall_curve(metrics['y_test'], metrics['y_prob'])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=rec, y=prec, mode='lines', line=dict(color='#00d4ff', width=2)))
            fig2.update_layout(xaxis_title='Recall', yaxis_title='Precision', template='plotly_dark',
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Classification Report</p>', unsafe_allow_html=True)
    rpt = metrics['report']
    rpt_df = pd.DataFrame({
        'Class':     ['Legitimate (0)', 'Fraudulent (1)', 'Macro Avg'],
        'Precision': [rpt['0']['precision'], rpt['1']['precision'], rpt['macro avg']['precision']],
        'Recall':    [rpt['0']['recall'],    rpt['1']['recall'],    rpt['macro avg']['recall']],
        'F1-Score':  [rpt['0']['f1-score'],  rpt['1']['f1-score'],  rpt['macro avg']['f1-score']],
        'Support':   [int(rpt['0']['support']), int(rpt['1']['support']),
                      int(rpt['0']['support'])+int(rpt['1']['support'])],
    })
    st.dataframe(rpt_df.style.format({'Precision':'{:.4f}','Recall':'{:.4f}','F1-Score':'{:.4f}'})
                 .background_gradient(cmap='Blues', subset=['Precision','Recall','F1-Score']),
                 use_container_width=True)


# ═══════════════════════════════════════════════
# DATASET EXPLORER
# ═══════════════════════════════════════════════
elif page == "📂 Dataset Explorer":
    st.markdown('<h2 style="color:#00d4ff;font-family:Space Mono">📂 Dataset Explorer</h2>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Fraud Cases",   f"{df['Class'].sum():,}")
    c3.metric("Features",      f"{len(df.columns)-1}")

    st.markdown('<p class="section-title">Sample Data</p>', unsafe_allow_html=True)
    n  = st.slider("Rows to display", 5, 100, 20)
    fc = st.selectbox("Filter by class", ["All","Legitimate (0)","Fraudulent (1)"])
    display_df = df.copy()
    if fc == "Legitimate (0)": display_df = df[df['Class']==0]
    if fc == "Fraudulent (1)": display_df = df[df['Class']==1]
    st.dataframe(display_df.head(n).style.applymap(
        lambda x: 'background-color:#3d0000;color:#ff6666' if x==1 else '', subset=['Class']
    ), use_container_width=True)

    st.markdown('<p class="section-title">Statistical Summary</p>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(3), use_container_width=True)
