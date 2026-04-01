import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
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

# ── Matplotlib global palette ────────────────────────────────────────────────
BG     = "#080c14"
PANEL  = "#0e1420"
BORDER = "#1c2a3a"
CYAN   = "#00e5ff"
GREEN  = "#00ff9d"
RED    = "#ff3366"
AMBER  = "#ffb300"
PURPLE = "#b388ff"
TEXT   = "#ffffff"
SUB    = "#ffffff"

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': PANEL,
    'axes.edgecolor': BORDER, 'axes.labelcolor': '#ffffff',
    'axes.titlecolor': '#ffffff', 'text.color': '#ffffff',
    'xtick.color': '#ffffff', 'ytick.color': '#ffffff',
    'grid.color': BORDER, 'grid.linestyle': '--', 'grid.alpha': 0.35,
    'legend.facecolor': PANEL, 'legend.edgecolor': BORDER, 'legend.fontsize': 8,
    'font.family': ['monospace'], 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield · ML Detection",
    page_icon="🛡️", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"]{font-family:'Exo 2',sans-serif;color:#ffffff}
.stApp{
  background:#080c14;
  background-image:
    radial-gradient(ellipse 80% 60% at 50% -10%,rgba(0,229,255,.07) 0%,transparent 70%),
    radial-gradient(ellipse 60% 50% at 90% 90%,rgba(0,255,157,.05) 0%,transparent 60%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#07111d 0%,#090f1a 100%);
  border-right:1px solid #1c2a3a;
}
section[data-testid="stSidebar"] .stRadio label{color:#ffffff!important;font-size:.82rem!important;transition:color .2s}
section[data-testid="stSidebar"] .stRadio label:hover{color:#00e5ff!important}
section[data-testid="stSidebar"] .stSelectbox label{color:#ffffff!important;font-size:.78rem!important}

.sb-brand{
  background:linear-gradient(135deg,#0a1929,#0d2137);
  border-bottom:1px solid #1c2a3a;
  padding:1.4rem 1.2rem 1.2rem;
  margin-bottom:.5rem;
}
.sb-icon{font-size:2rem;line-height:1}
.sb-name{
  font-family:'Share Tech Mono',monospace;font-size:1.25rem;
  color:#00e5ff;letter-spacing:3px;
  text-shadow:0 0 16px rgba(0,229,255,.5);margin-top:.3rem;
}
.sb-sub{font-size:.65rem;color:#ffffff;letter-spacing:1px;margin-top:.1rem}
.sb-nav-lbl{font-size:.6rem;color:#ffffff;text-transform:uppercase;letter-spacing:2px;padding:0 1rem;margin:.5rem 0 .3rem}
.sb-footer{padding:0 .5rem;font-size:.62rem;color:#ffffff;line-height:2}

/* ── Hero ── */
.hero{
  position:relative;overflow:hidden;
  background:linear-gradient(135deg,#071020 0%,#0a1830 50%,#071525 100%);
  border:1px solid #1c2a3a;border-radius:20px;
  padding:clamp(1.5rem,4vw,2.8rem) clamp(1.2rem,3vw,2.5rem);
  margin-bottom:1.5rem;
  box-shadow:0 0 60px rgba(0,229,255,.06),inset 0 1px 0 rgba(0,229,255,.1);
}
.hero::before{
  content:'';position:absolute;inset:0;
  background:
    repeating-linear-gradient(0deg,transparent,transparent 24px,rgba(0,229,255,.015) 24px,rgba(0,229,255,.015) 25px),
    repeating-linear-gradient(90deg,transparent,transparent 24px,rgba(0,229,255,.015) 24px,rgba(0,229,255,.015) 25px);
  border-radius:20px;pointer-events:none;
}
.hero-inner{position:relative;z-index:1;text-align:center}
.hero-badge{
  display:inline-block;
  background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.25);
  border-radius:100px;padding:.25rem 1rem;
  font-size:.68rem;color:#00e5ff;letter-spacing:3px;text-transform:uppercase;
  font-family:'Share Tech Mono',monospace;margin-bottom:.8rem;
}
.hero h1{
  font-family:'Share Tech Mono',monospace;
  font-size:clamp(1.6rem,6vw,3rem);
  color:#00e5ff;letter-spacing:4px;
  text-shadow:0 0 30px rgba(0,229,255,.4),0 0 60px rgba(0,229,255,.15);
  line-height:1.1;
}
.hero-sub{color:#ffffff;font-size:clamp(.75rem,2vw,.9rem);margin-top:.5rem;letter-spacing:1px}
.hero-stats{display:flex;gap:clamp(.8rem,3vw,2rem);justify-content:center;flex-wrap:wrap;margin-top:1.5rem}
.hero-stat{text-align:center}
.hero-stat-val{
  font-family:'Share Tech Mono',monospace;font-size:clamp(1.1rem,3.5vw,1.6rem);
  color:#00ff9d;text-shadow:0 0 12px rgba(0,255,157,.4);
}
.hero-stat-lbl{font-size:.63rem;color:#ffffff;text-transform:uppercase;letter-spacing:1.5px;margin-top:.1rem}
.hero-div{width:1px;background:#1c2a3a;height:40px;align-self:center}

/* ── Metric card ── */
.mcard{
  background:linear-gradient(145deg,#0c1520,#0f1d2e);
  border:1px solid #1c2a3a;border-radius:14px;
  padding:clamp(.9rem,2.5vw,1.3rem) clamp(.8rem,2vw,1.2rem);
  position:relative;overflow:hidden;
  transition:transform .25s,box-shadow .25s;
  margin-bottom:.5rem;
}
.mcard::after{
  content:'';position:absolute;top:0;left:0;right:0;
  height:2px;border-radius:14px 14px 0 0;
}
.mcard.cyan::after {background:linear-gradient(90deg,transparent,#00e5ff,transparent)}
.mcard.green::after{background:linear-gradient(90deg,transparent,#00ff9d,transparent)}
.mcard.red::after  {background:linear-gradient(90deg,transparent,#ff3366,transparent)}
.mcard.amber::after{background:linear-gradient(90deg,transparent,#ffb300,transparent)}
.mcard:hover{transform:translateY(-4px);box-shadow:0 8px 32px rgba(0,229,255,.08)}
.mcard-icon{font-size:1.2rem;margin-bottom:.35rem;opacity:.85}
.mcard-lbl{font-size:.62rem;color:#ffffff;text-transform:uppercase;letter-spacing:2px}
.mcard-val{
  font-family:'Share Tech Mono',monospace;
  font-size:clamp(1.2rem,3.5vw,1.8rem);font-weight:700;margin-top:.2rem;line-height:1;
}
.mcard.cyan  .mcard-val{color:#00e5ff;text-shadow:0 0 12px rgba(0,229,255,.35)}
.mcard.green .mcard-val{color:#00ff9d;text-shadow:0 0 12px rgba(0,255,157,.35)}
.mcard.red   .mcard-val{color:#ff3366;text-shadow:0 0 12px rgba(255,51,102,.35)}
.mcard.amber .mcard-val{color:#ffb300;text-shadow:0 0 12px rgba(255,179,0,.35)}

/* ── Section header ── */
.sec-hdr{display:flex;align-items:center;gap:.6rem;margin-bottom:.9rem}
.sec-line{flex:1;height:1px;background:linear-gradient(90deg,#1c2a3a,transparent)}
.sec-txt{
  font-family:'Share Tech Mono',monospace;font-size:.68rem;
  color:#00e5ff;text-transform:uppercase;letter-spacing:3px;white-space:nowrap;
}

/* ── Chart panel ── */
.cpanel{
  background:linear-gradient(145deg,#0c1520,#0f1d2e);
  border:1px solid #1c2a3a;border-radius:14px;
  padding:clamp(.8rem,2vw,1.2rem);margin-bottom:1rem;
  transition:box-shadow .25s;
}
.cpanel:hover{box-shadow:0 4px 24px rgba(0,229,255,.06)}

/* ── Info box ── */
.info-box{
  background:linear-gradient(145deg,#071525,#0a1e35);
  border-left:3px solid #00e5ff;border-radius:0 10px 10px 0;
  padding:clamp(.8rem,2vw,1.1rem) clamp(.9rem,2.5vw,1.3rem);
  margin:1rem 0;font-size:clamp(.78rem,2vw,.88rem);color:#ffffff;line-height:1.6;
}
.info-box strong{color:#00e5ff}
.info-box em{color:#ffffff}

/* ── Page title ── */
.page-title{
  font-family:'Share Tech Mono',monospace;
  font-size:clamp(1.1rem,4vw,1.6rem);letter-spacing:3px;
  color:#ffffff;
  margin-bottom:1.2rem;padding-bottom:.7rem;
  border-bottom:1px solid #1c2a3a;
  display:flex;align-items:center;gap:.6rem;
}

/* ── SMOTE cards ── */
.smote-c{
  background:linear-gradient(145deg,#091a0f,#0d2217);
  border:1px solid rgba(0,255,157,.18);border-radius:12px;
  padding:clamp(.8rem,2vw,1.1rem);text-align:center;margin-bottom:.5rem;
}
.smote-c-lbl{font-size:.62rem;color:#ffffff;text-transform:uppercase;letter-spacing:2px}
.smote-c-val{
  font-family:'Share Tech Mono',monospace;
  font-size:clamp(1.2rem,3.5vw,1.8rem);
  color:#00ff9d;text-shadow:0 0 10px rgba(0,255,157,.35);margin-top:.2rem;
}

/* ── Variant grid ── */
.var-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,200px),1fr));gap:.75rem;margin-top:.5rem}
.var-card{background:linear-gradient(145deg,#0c1520,#0f1d2e);border-radius:12px;padding:clamp(.8rem,2vw,1rem)}
.var-name{font-family:'Share Tech Mono',monospace;font-size:.76rem;font-weight:700;margin-bottom:.45rem}
.var-desc{font-size:.78rem;color:#ffffff;line-height:1.5}

/* ── Compare grid ── */
.cmp-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,170px),1fr));gap:.75rem;margin-bottom:1.2rem}
.cmp-card{
  background:linear-gradient(145deg,#0c1520,#0f1d2e);
  border:1px solid #1c2a3a;border-radius:12px;
  padding:clamp(.9rem,2vw,1.1rem);text-align:center;
}
.cmp-lbl{font-size:.62rem;color:#ffffff;text-transform:uppercase;letter-spacing:2px;margin-bottom:.6rem}
.cmp-row{display:flex;align-items:center;justify-content:space-around;gap:.4rem;flex-wrap:wrap}
.cmp-val{font-family:'Share Tech Mono',monospace;font-size:1.2rem}
.cmp-arrow{color:#ffffff;font-size:1rem}
.cmp-delta{font-size:.7rem;font-family:'Share Tech Mono',monospace;margin-top:.5rem}
.cmp-sublbl{font-size:.62rem;color:#ffffff;margin-bottom:.15rem}

/* ── Predict ── */
.pred-wrap{
  background:linear-gradient(145deg,#0c1520,#0f1d2e);
  border:1px solid #1c2a3a;border-radius:16px;
  padding:clamp(1rem,3vw,1.5rem);margin-bottom:1rem;
}
.fraud-alert{
  background:linear-gradient(135deg,#1a0010,#2a0018);
  border:1px solid rgba(255,51,102,.4);border-radius:16px;
  padding:clamp(1.2rem,3vw,2rem);text-align:center;
  box-shadow:0 0 40px rgba(255,51,102,.12),inset 0 1px 0 rgba(255,51,102,.15);
  animation:pulse-r 2.5s ease-in-out infinite;
}
@keyframes pulse-r{
  0%,100%{box-shadow:0 0 30px rgba(255,51,102,.1)}
  50%    {box-shadow:0 0 60px rgba(255,51,102,.25)}
}
.legit-badge{
  background:linear-gradient(135deg,#001a0e,#00291a);
  border:1px solid rgba(0,255,157,.3);border-radius:16px;
  padding:clamp(1.2rem,3vw,2rem);text-align:center;
  box-shadow:0 0 40px rgba(0,255,157,.08);
}
.a-title{font-family:'Share Tech Mono',monospace;font-size:clamp(1rem,3.5vw,1.5rem);letter-spacing:2px;margin-bottom:.5rem}
.a-prob{font-family:'Share Tech Mono',monospace;font-size:clamp(.85rem,2.5vw,1.1rem)}
.a-hint{font-size:.8rem;color:#ffffff;margin-top:.3rem}

/* ── Streamlit overrides ── */
.stButton>button{
  width:100%;
  background:linear-gradient(135deg,#0a1d35,#0c2545);
  color:#00e5ff;border:1px solid rgba(0,229,255,.3);
  border-radius:10px;font-family:'Share Tech Mono',monospace;
  font-size:.88rem;letter-spacing:2px;padding:.65rem 1.2rem;transition:all .2s;
}
.stButton>button:hover{
  background:linear-gradient(135deg,#0d2545,#10306a);
  border-color:#00e5ff;box-shadow:0 0 20px rgba(0,229,255,.2);color:#fff;
}
div[data-testid="stDataFrame"]{border:1px solid #1c2a3a;border-radius:10px}
.stMetric{background:linear-gradient(145deg,#0c1520,#0f1d2e);border:1px solid #1c2a3a;border-radius:12px;padding:.8rem 1rem}
.stMetric label{color:#ffffff!important;font-size:.7rem!important;letter-spacing:1.5px;text-transform:uppercase}
.stMetric [data-testid="stMetricValue"]{font-family:'Share Tech Mono',monospace!important;color:#00e5ff!important}

/* ── Global text overrides — force all remaining text white ── */
p, span, div, label, li, td, th, small, h1, h2, h3, h4, h5, h6 {color:#ffffff}
.stSlider label {color:#ffffff!important}
.stSlider [data-testid="stWidgetLabel"] {color:#ffffff!important}
[data-testid="stWidgetLabel"] {color:#ffffff!important}
[data-testid="stMarkdownContainer"] p {color:#ffffff!important}
[data-testid="stMarkdownContainer"] li {color:#ffffff!important}
.stRadio [data-testid="stWidgetLabel"] {color:#ffffff!important}
.stNumberInput label {color:#ffffff!important}
/* Sidebar — white everything EXCEPT selectbox dropdown elements */
section[data-testid="stSidebar"] *:not([data-baseweb="select"] *):not([data-baseweb="popover"] *):not([data-baseweb="menu"] *) {color:#ffffff!important}
section[data-testid="stSidebar"] .sb-name {color:#00e5ff!important}
section[data-testid="stSidebar"] .sb-nav-lbl {color:#ffffff!important}
section[data-testid="stSidebar"] .stRadio label:hover {color:#00e5ff!important}

/* ── Dropdown black text — all screen sizes, sidebar + main ────────────────
   Declared last so they win over every white rule above regardless of device
   ────────────────────────────────────────────────────────────────────────── */
[data-baseweb="select"] *,
[data-baseweb="select"] div,
[data-baseweb="select"] span,
[data-baseweb="select"] input,
[data-baseweb="select"] div[class*="ValueContainer"],
[data-baseweb="select"] div[class*="ValueContainer"] *,
[data-baseweb="select"] div[class*="singleValue"],
[data-baseweb="select"] div[class*="placeholder"],
.stSelectbox [data-baseweb="select"] div,
.stSelectbox [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] div[class*="ValueContainer"] *,
section[data-testid="stSidebar"] [data-baseweb="select"] div[class*="singleValue"],
section[data-testid="stSidebar"] [data-baseweb="select"] div[class*="placeholder"] {color:#000000!important}
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="popover"] div,
[data-baseweb="popover"] span,
[data-baseweb="popover"] li,
[data-baseweb="popover"] ul li span {background-color:#ffffff;color:#000000!important}
[data-baseweb="menu"],
[data-baseweb="menu"] *,
[data-baseweb="menu"] div,
[data-baseweb="menu"] span,
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"],
[data-baseweb="menu"] [role="option"] * {background-color:#ffffff;color:#000000!important}
[data-baseweb="menu"] [aria-selected="true"],
[data-baseweb="menu"] [aria-selected="true"] * {background-color:#e8f4ff!important;color:#000000!important}
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"]:hover *,
[data-baseweb="menu"] [role="option"]:active,
[data-baseweb="menu"] [role="option"]:active * {background-color:#f0f0f0!important;color:#000000!important}

/* ── Responsive ── */
@media(max-width:640px){
  .hero-stats{flex-direction:column;gap:.6rem}
  .hero-div{display:none}
  [data-testid="column"]{min-width:100%!important}
  .cmp-grid{grid-template-columns:1fr}
}
@media(min-width:641px) and (max-width:1024px){
  .cmp-grid{grid-template-columns:repeat(2,1fr)}
}
</style>
""", unsafe_allow_html=True)


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _fig(w=7, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax

def styled(fig):
    fig.patch.set_facecolor(BG)
    fig.tight_layout(pad=1.2)
    return fig

def plot_histogram(df):
    fig, ax = _fig()
    for cls, col, lbl in [(0,CYAN,'Legitimate'),(1,RED,'Fraudulent')]:
        d = df[df['Class']==cls]['Amount']
        ax.hist(d, bins=55, alpha=0.72, color=col, label=lbl, density=True, edgecolor='none')
    ax.set_xlabel('Amount ($)'); ax.set_ylabel('Density')
    ax.legend(); ax.grid(True, alpha=0.3)
    return styled(fig)

def plot_hourly(df):
    fig, ax = _fig()
    hourly = df.groupby(['Hour','Class']).size().reset_index(name='Count')
    for cls, col, lbl in [(0,CYAN,'Legitimate'),(1,RED,'Fraudulent')]:
        d = hourly[hourly['Class']==cls]
        ax.fill_between(d['Hour'], d['Count'], alpha=0.1, color=col)
        ax.plot(d['Hour'], d['Count'], color=col, marker='o', ms=4,
                label=lbl, lw=2, markerfacecolor=BG, markeredgewidth=1.5)
    ax.set_xlabel('Hour of Day'); ax.set_ylabel('Transactions')
    ax.set_xticks(range(0,24,2)); ax.legend(); ax.grid(True, alpha=0.3)
    return styled(fig)

def plot_donut(df):
    fig, ax = _fig(4.5, 3.8)
    counts = df['Class'].value_counts()
    sizes  = [counts.get(0,0), counts.get(1,0)]
    ax.pie(sizes, colors=[CYAN, RED], autopct='%1.1f%%', startangle=90,
           pctdistance=0.78,
           wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=3))
    total = sum(sizes)
    ax.text(0, 0.08, f'{total:,}', ha='center', va='center',
            fontfamily='monospace', fontsize=15, color='#ffffff', fontweight='bold')
    ax.text(0, -0.22, 'TOTAL', ha='center', va='center',
            fontsize=7, color='#ffffff', fontfamily='monospace')
    handles = [mpatches.Patch(color=CYAN, label='Legitimate'),
               mpatches.Patch(color=RED,  label='Fraudulent')]
    ax.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.05), fontsize=8)
    return styled(fig)

def plot_roc(m1, m2, l1='SMOTE', l2='No-SMOTE'):
    fig, ax = _fig()
    for m, col, lbl, ls in [(m1,GREEN,l1,'-'),(m2,CYAN,l2,'--')]:
        fpr, tpr, _ = roc_curve(m['y_test'], m['y_prob'])
        ax.plot(fpr, tpr, color=col, lw=2, ls=ls, label=f'{lbl}  AUC={m["roc_auc"]:.3f}')
        if ls=='-': ax.fill_between(fpr, tpr, alpha=0.07, color=col)
    ax.plot([0,1],[0,1], color=BORDER, ls=':', lw=1.2, label='Random')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    return styled(fig)

def plot_smote_bar(before, after):
    fig, ax = _fig()
    x  = np.arange(2); w = 0.32
    bv = [before.get(0,0), before.get(1,0)]
    av = [after.get(0,0),  after.get(1,0)]
    b1 = ax.bar(x-w/2, bv, w, label='Before', color=CYAN,  alpha=0.82, zorder=3)
    b2 = ax.bar(x+w/2, av, w, label='After',  color=GREEN, alpha=0.82, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(['Legitimate','Fraudulent'])
    ax.legend(); ax.grid(True, axis='y', alpha=0.3, zorder=0)
    for bar, v, c in [(b, vv, cc) for b,vv,cc in list(zip(b1,bv,[CYAN]*2))+list(zip(b2,av,[GREEN]*2))]:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                f'{v:,}', ha='center', fontsize=7.5, color=c)
    return styled(fig)

def plot_ratio(before, after):
    fig, ax = _fig(4.5, 3.8)
    rb = before.get(1,0)/max(sum(before.values()),1)*100
    ra = after.get(1,0) /max(sum(after.values()), 1)*100
    bars = ax.bar(['Before\nSMOTE','After\nSMOTE'], [rb,ra],
                  color=[RED,GREEN], alpha=0.85, width=0.45, zorder=3)
    ax.set_ylabel('Fraud %'); ax.set_ylim(0, max(ra,rb)*1.7)
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    for bar, v, c in zip(bars,[rb,ra],[RED,GREEN]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                f'{v:.1f}%', ha='center', fontsize=11, color=c, fontweight='bold')
    return styled(fig)

def plot_pca(X_res, y_res, X_orig):
    n = len(X_orig)
    lmap = {(True,0):'Legitimate',(True,1):'Original Fraud',(False,1):'Synthetic Fraud'}
    labels = [lmap.get((i<n, cls),'Synthetic Fraud') for i,cls in enumerate(y_res)]
    pca    = PCA(n_components=2, random_state=42)
    c      = pca.fit_transform(X_res)
    df_p   = pd.DataFrame({'PC1':c[:,0],'PC2':c[:,1],'Type':labels})
    pal    = {'Legitimate':(CYAN,.2),'Original Fraud':(RED,.7),'Synthetic Fraud':(GREEN,.55)}
    fig, ax = _fig(8, 4.2)
    handles = []
    for t in ['Legitimate','Original Fraud','Synthetic Fraud']:
        sub = df_p[df_p['Type']==t]
        col, alp = pal[t]
        ax.scatter(sub['PC1'], sub['PC2'], c=col, s=1.8, alpha=alp, rasterized=True)
        handles.append(mpatches.Patch(color=col, label=t))
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.25)
    return styled(fig)

def plot_metrics_bar(m_no, m_sm):
    lbls   = ['Accuracy','F1','ROC-AUC','Recall','Precision']
    n_vals = [m_no['accuracy'],m_no['f1'],m_no['roc_auc'],
              m_no['report']['1']['recall'],m_no['report']['1']['precision']]
    s_vals = [m_sm['accuracy'],m_sm['f1'],m_sm['roc_auc'],
              m_sm['report']['1']['recall'],m_sm['report']['1']['precision']]
    x = np.arange(len(lbls)); w = 0.33
    fig, ax = _fig(8,4)
    b1 = ax.bar(x-w/2, n_vals, w, label='No SMOTE',   color=CYAN,  alpha=0.82, zorder=3)
    b2 = ax.bar(x+w/2, s_vals, w, label='With SMOTE', color=GREEN, alpha=0.82, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=8)
    ax.set_ylim(0,1.15); ax.legend(); ax.grid(True, axis='y', alpha=0.3, zorder=0)
    for bar, v, c in [(b,vv,cc) for b,vv,cc in list(zip(b1,n_vals,[CYAN]*5))+list(zip(b2,s_vals,[GREEN]*5))]:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                f'{v:.2f}', ha='center', fontsize=6.5, color=c)
    return styled(fig)

def plot_cm(cm, title=''):
    fig, ax = _fig(4.2, 3.5)
    cmap = LinearSegmentedColormap.from_list('fs', ['#0a1525','#0046aa','#00e5ff'], N=256)
    ax.imshow(cm, cmap=cmap, aspect='auto', vmin=0)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred Legit','Pred Fraud'], fontsize=8)
    ax.set_yticklabels(['Act Legit','Act Fraud'],   fontsize=8)
    cmap2 = [[GREEN,RED],[AMBER,GREEN]]
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',
                    fontsize=15,color=cmap2[i][j],fontweight='bold',fontfamily='monospace')
    if title: ax.set_title(title, color='#ffffff', fontsize=9, pad=8)
    return styled(fig)

def plot_fi(features, importances):
    idx  = np.argsort(importances)
    vals = importances[idx]
    fts  = [features[i] for i in idx]
    fig, ax = _fig(6,4)
    colors = [CYAN if v<np.percentile(vals,60) else GREEN for v in vals]
    bars = ax.barh(fts, vals, color=colors, alpha=0.85, zorder=3)
    ax.set_xlabel('Importance'); ax.grid(True, axis='x', alpha=0.3, zorder=0)
    for bar, v in zip(bars, vals):
        ax.text(v+.001, bar.get_y()+bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=7.5, color='#ffffff')
    return styled(fig)

def plot_pr(metrics):
    prec, rec, _ = precision_recall_curve(metrics['y_test'], metrics['y_prob'])
    fig, ax = _fig()
    ax.plot(rec, prec, color=CYAN, lw=2, zorder=3)
    ax.fill_between(rec, prec, alpha=0.08, color=CYAN)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.grid(True, alpha=0.3)
    return styled(fig)

def plot_pred_bar(prob):
    fig, ax = _fig(5.5, 3.2)
    bars = ax.bar(['LEGITIMATE','FRAUDULENT'], prob,
                  color=[GREEN,RED], alpha=0.85, width=0.4, zorder=3)
    ax.set_ylim(0,1.22); ax.set_ylabel('Probability')
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    for bar, v, c in zip(bars, prob, [GREEN,RED]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                f'{v:.1%}', ha='center', fontsize=13, color=c,
                fontweight='bold', fontfamily='monospace')
    ax.set_xticklabels(['LEGITIMATE','FRAUDULENT'], fontsize=9, fontfamily='monospace')
    ax.set_title('Prediction Confidence', color='#ffffff', fontsize=10, pad=8)
    return styled(fig)


# ── Data & training ───────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=10000):
    np.random.seed(42)
    nf = int(n*.02); nl = n-nf
    legit = pd.DataFrame({
        'Amount':np.random.exponential(80,nl),'Hour':np.random.randint(0,24,nl),
        'V1':np.random.normal(0,1,nl),'V2':np.random.normal(0,1,nl),
        'V3':np.random.normal(0,1,nl),'V4':np.random.normal(0,1,nl),
        'V5':np.random.normal(0,1,nl),'V6':np.random.normal(0,1,nl),
        'V7':np.random.normal(0,1,nl),'V8':np.random.normal(0,1,nl),'Class':0})
    fraud = pd.DataFrame({
        'Amount':np.random.exponential(200,nf),'Hour':np.random.choice([0,1,2,3,22,23],nf),
        'V1':np.random.normal(-3,1.5,nf),'V2':np.random.normal(2,1.5,nf),
        'V3':np.random.normal(-2.5,1,nf),'V4':np.random.normal(2.5,1,nf),
        'V5':np.random.normal(-1.5,1,nf),'V6':np.random.normal(1.5,1,nf),
        'V7':np.random.normal(-2,1,nf),  'V8':np.random.normal(2,1,nf),'Class':1})
    df = pd.concat([legit,fraud],ignore_index=True).sample(frac=1,random_state=42).reset_index(drop=True)
    df['Amount'] = df['Amount'].clip(.5,5000)
    return df

FEATURES = ['Amount','Hour','V1','V2','V3','V4','V5','V6','V7','V8']
SMOTE_VARIANTS = {
    "SMOTE (Standard)": SMOTE(random_state=42),
    "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
    "SVM-SMOTE":        SVMSMOTE(random_state=42),
}

def _fit(X_tr, y_tr, X_te, y_te, mtype):
    clf = (RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,class_weight='balanced')
           if mtype=="Random Forest"
           else LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te); ypr = clf.predict_proba(X_te)[:,1]
    return clf, {"accuracy":accuracy_score(y_te,yp),"f1":f1_score(y_te,yp),
                 "roc_auc":roc_auc_score(y_te,ypr),"confusion":confusion_matrix(y_te,yp),
                 "report":classification_report(y_te,yp,output_dict=True),
                 "y_test":y_te,"y_prob":ypr,
                 "feature_importance":clf.feature_importances_ if mtype=="Random Forest" else None}

@st.cache_resource
def train_all(mtype, smv):
    df = generate_dataset()
    X  = df[FEATURES].values; y = df['Class'].values
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=.2,random_state=42,stratify=y)
    _, m_no = _fit(Xtr,ytr,Xte,yte,mtype)
    Xr,yr   = SMOTE_VARIANTS[smv].fit_resample(Xtr,ytr)
    clf_sm, m_sm = _fit(Xr,yr,Xte,yte,mtype)
    bef = dict(zip(*np.unique(ytr,return_counts=True)))
    aft = dict(zip(*np.unique(yr, return_counts=True)))
    return sc, clf_sm, m_no, m_sm, bef, aft, Xtr, Xr, yr


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-icon">🛡️</div>
        <div class="sb-name">FRAUDSHIELD</div>
        <div class="sb-sub">ML DETECTION SYSTEM v2.0</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="sb-nav-lbl">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "🏠  Dashboard","🔬  SMOTE Analysis","⚖️  SMOTE vs No-SMOTE",
        "🔍  Predict Transaction","📊  Model Insights","📂  Dataset Explorer",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="sb-nav-lbl">Configuration</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Classifier", ["Random Forest","Logistic Regression"])
    smote_choice = st.selectbox("SMOTE Variant", list(SMOTE_VARIANTS.keys()))
    st.markdown("---")
    st.markdown('<div class="sb-footer">📦 streamlit · sklearn<br>🧬 imbalanced-learn<br>📊 matplotlib · pandas</div>',
                unsafe_allow_html=True)

df = generate_dataset()
scaler, model, metrics_no, metrics_sm, before_counts, after_counts, X_orig, X_res, y_res = \
    train_all(model_choice, smote_choice)
metrics = metrics_sm

# ── UI helpers ────────────────────────────────────────────────────────────────
def sec(title):
    st.markdown(f'<div class="sec-hdr"><span class="sec-txt">{title}</span><div class="sec-line"></div></div>',
                unsafe_allow_html=True)

def mcard(val, lbl, accent='cyan', icon=''):
    st.markdown(f'<div class="mcard {accent}"><div class="mcard-icon">{icon}</div>'
                f'<div class="mcard-lbl">{lbl}</div><div class="mcard-val">{val}</div></div>',
                unsafe_allow_html=True)

def cpanel(fig):
    st.markdown('<div class="cpanel">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close('all')


# ═══════════════════════════════════════════════════════════════════════════════
#  🏠  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    rpt = metrics['report']
    st.markdown(f"""
    <div class="hero"><div class="hero-inner">
      <div class="hero-badge">⬡ REAL-TIME FRAUD INTELLIGENCE</div>
      <h1>🛡️ FRAUD SHIELD</h1>
      <p class="hero-sub">{model_choice} · {smote_choice} · SMOTE-Balanced Pipeline</p>
      <div class="hero-stats">
        <div class="hero-stat"><div class="hero-stat-val">{metrics['accuracy']:.1%}</div><div class="hero-stat-lbl">Accuracy</div></div>
        <div class="hero-div"></div>
        <div class="hero-stat"><div class="hero-stat-val">{metrics['roc_auc']:.3f}</div><div class="hero-stat-lbl">ROC-AUC</div></div>
        <div class="hero-div"></div>
        <div class="hero-stat"><div class="hero-stat-val">{metrics['f1']:.3f}</div><div class="hero-stat-lbl">F1 Score</div></div>
        <div class="hero-div"></div>
        <div class="hero-stat"><div class="hero-stat-val">{df['Class'].mean()*100:.1f}%</div><div class="hero-stat-lbl">Fraud Rate</div></div>
      </div>
    </div></div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: mcard(f"{len(df):,}",                       "Total Transactions",  "cyan",  "💳")
    with c2: mcard(f"{df['Class'].sum():,}",              "Fraud Cases Detected","red",   "⚠️")
    with c3: mcard(f"{rpt['1']['recall']:.1%}",           "Fraud Recall",        "green", "🎯")
    with c4: mcard(f"{rpt['1']['precision']:.1%}",        "Fraud Precision",     "amber", "🔎")

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: sec("Amount Distribution"); cpanel(plot_histogram(df))
    with c2: sec("Hourly Pattern");      cpanel(plot_hourly(df))
    c3,c4 = st.columns(2)
    with c3: sec("Class Balance");       cpanel(plot_donut(df))
    with c4: sec("ROC Curve Comparison");cpanel(plot_roc(metrics, metrics_no))


# ═══════════════════════════════════════════════════════════════════════════════
#  🔬  SMOTE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "SMOTE Analysis" in page:
    st.markdown('<div class="page-title"><span style="color:#00ff9d">🔬</span> SMOTE ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
      <strong>Synthetic Minority Over-sampling Technique</strong> — creates synthetic fraud samples
      by interpolating between existing minority-class instances in feature space. It doesn't
      duplicate — it <em>manufactures</em> brand-new plausible examples, forcing the model
      to learn far more robust fraud decision boundaries.<br><br>
      <strong>Active variant:</strong> {smote_choice}
    </div>""", unsafe_allow_html=True)

    sec("Class Balance — Before vs After SMOTE")
    c1,c2,c3,c4 = st.columns(4)
    with c1: mcard(f"{before_counts.get(0,0):,}", "Legit · Before", "cyan",  "")
    with c2: mcard(f"{before_counts.get(1,0):,}", "Fraud · Before", "red",   "")
    with c3: st.markdown(f'<div class="smote-c"><div class="smote-c-lbl">Legit · After</div><div class="smote-c-val">{after_counts.get(0,0):,}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="smote-c"><div class="smote-c-lbl">Fraud · After</div><div class="smote-c-val">{after_counts.get(1,0):,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: sec("Sample Counts");        cpanel(plot_smote_bar(before_counts, after_counts))
    with c2: sec("Fraud Ratio Shift");    cpanel(plot_ratio(before_counts, after_counts))

    sec("PCA 2D Projection — Original vs Synthetic")
    st.markdown('<p style="color:#ffffff;font-size:.74rem;margin-bottom:.5rem">🟢 Synthetic Fraud &nbsp; 🔴 Original Fraud &nbsp; 🔵 Legitimate</p>', unsafe_allow_html=True)
    cpanel(plot_pca(X_res, y_res, X_orig))

    sec("SMOTE Variant Guide")
    descs = [
        ("SMOTE (Standard)", CYAN,   "Interpolates between a minority sample and one of its k nearest neighbours. Best general-purpose choice."),
        ("Borderline-SMOTE", AMBER,  "Focuses synthesis only on minority samples near the decision boundary — hardest classification zone."),
        ("SVM-SMOTE",        PURPLE, "Uses SVMs to detect high-risk boundary regions and generates samples aligned with the SVM margin."),
    ]
    st.markdown('<div class="var-grid">', unsafe_allow_html=True)
    for name, col, desc in descs:
        border = f"2px solid {col}" if name==smote_choice else f"1px solid {BORDER}"
        active = "▶ " if name==smote_choice else ""
        st.markdown(f'<div class="var-card" style="border:{border}"><div class="var-name" style="color:{col}">{active}{name}</div><div class="var-desc">{desc}</div></div>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ⚖️  SMOTE vs NO-SMOTE
# ═══════════════════════════════════════════════════════════════════════════════
elif "No-SMOTE" in page:
    st.markdown('<div class="page-title"><span style="color:#00e5ff">⚖️</span> SMOTE vs NO-SMOTE COMPARISON</div>', unsafe_allow_html=True)

    pairs = [
        ("F1 Score",        "📊", metrics_no['f1'],                     metrics_sm['f1']),
        ("ROC-AUC",         "📈", metrics_no['roc_auc'],                metrics_sm['roc_auc']),
        ("Fraud Recall",    "🎯", metrics_no['report']['1']['recall'],  metrics_sm['report']['1']['recall']),
        ("Fraud Precision", "🔎", metrics_no['report']['1']['precision'],metrics_sm['report']['1']['precision']),
    ]
    st.markdown('<div class="cmp-grid">', unsafe_allow_html=True)
    for lbl, icon, nv, sv in pairs:
        better = sv>=nv; d = sv-nv; sign="▲" if d>=0 else "▼"; dc=GREEN if d>=0 else RED
        st.markdown(f"""
        <div class="cmp-card">
          <div class="cmp-lbl">{icon} {lbl}</div>
          <div class="cmp-row">
            <div><div class="cmp-sublbl">No SMOTE</div>
                 <div class="cmp-val" style="color:{CYAN}">{nv:.3f}</div></div>
            <div class="cmp-arrow">→</div>
            <div><div class="cmp-sublbl">With SMOTE</div>
                 <div class="cmp-val" style="color:{'#00ff9d' if better else RED}">{sv:.3f}</div></div>
          </div>
          <div class="cmp-delta" style="color:{dc}">{sign} {abs(d):.3f}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1: sec("All Metrics Side-by-Side"); cpanel(plot_metrics_bar(metrics_no, metrics_sm))
    with c2: sec("ROC Curve Overlay");        cpanel(plot_roc(metrics_sm, metrics_no, 'With SMOTE','No SMOTE'))

    sec("Confusion Matrices")
    c3,c4 = st.columns(2)
    with c3: cpanel(plot_cm(metrics_no['confusion'], 'No SMOTE'))
    with c4: cpanel(plot_cm(metrics_sm['confusion'], 'With SMOTE'))


# ═══════════════════════════════════════════════════════════════════════════════
#  🔍  PREDICT TRANSACTION
# ═══════════════════════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown('<div class="page-title"><span style="color:#00e5ff">🔍</span> PREDICT TRANSACTION</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#ffffff;font-size:.8rem;margin-bottom:1rem">Model: <span style="color:{CYAN}">{model_choice}</span> &nbsp;·&nbsp; SMOTE: <span style="color:{GREEN}">{smote_choice}</span></p>', unsafe_allow_html=True)

    st.markdown('<div class="pred-wrap">', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f'<p style="color:{CYAN};font-family:\'Share Tech Mono\',monospace;font-size:.72rem;letter-spacing:2px;margin-bottom:.4rem">TRANSACTION</p>', unsafe_allow_html=True)
        amount = st.number_input("Amount ($)", min_value=0.01, max_value=5000.0, value=150.0, step=0.01)
        hour   = st.slider("Hour", 0, 23, 14)
        v1     = st.slider("V1", -5.0, 5.0, 0.0, 0.01)
    with c2:
        st.markdown(f'<p style="color:{CYAN};font-family:\'Share Tech Mono\',monospace;font-size:.72rem;letter-spacing:2px;margin-bottom:.4rem">VECTORS A</p>', unsafe_allow_html=True)
        v2 = st.slider("V2", -5.0, 5.0, 0.0, 0.01)
        v3 = st.slider("V3", -5.0, 5.0, 0.0, 0.01)
        v4 = st.slider("V4", -5.0, 5.0, 0.0, 0.01)
    with c3:
        st.markdown(f'<p style="color:{CYAN};font-family:\'Share Tech Mono\',monospace;font-size:.72rem;letter-spacing:2px;margin-bottom:.4rem">VECTORS B</p>', unsafe_allow_html=True)
        v5 = st.slider("V5", -5.0, 5.0, 0.0, 0.01)
        v6 = st.slider("V6", -5.0, 5.0, 0.0, 0.01)
        v7 = st.slider("V7", -5.0, 5.0, 0.0, 0.01)
        v8 = st.slider("V8", -5.0, 5.0, 0.0, 0.01)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⬡  ANALYZE TRANSACTION", use_container_width=True):
        inp  = scaler.transform(np.array([[amount,hour,v1,v2,v3,v4,v5,v6,v7,v8]]))
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0]
        if pred==1:
            st.markdown(f"""
            <div class="fraud-alert">
              <div class="a-title" style="color:#ff3366">⚠ FRAUDULENT TRANSACTION DETECTED</div>
              <div class="a-prob" style="color:#ffffff">Fraud Probability: {prob[1]:.1%}</div>
              <div class="a-hint">Recommended: Block &amp; alert cardholder immediately</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-badge">
              <div class="a-title" style="color:#00ff9d">✓ LEGITIMATE TRANSACTION</div>
              <div class="a-prob" style="color:#ffffff">Legitimate Probability: {prob[0]:.1%}</div>
              <div class="a-hint">No anomalies detected — transaction approved</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        sec("Prediction Confidence")
        cpanel(plot_pred_bar(prob))


# ═══════════════════════════════════════════════════════════════════════════════
#  📊  MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Insights" in page:
    st.markdown('<div class="page-title"><span style="color:#00e5ff">📊</span> MODEL INSIGHTS</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#ffffff;font-size:.8rem;margin-bottom:1rem">{model_choice} &nbsp;·&nbsp; {smote_choice}</p>', unsafe_allow_html=True)
    rpt = metrics['report']
    c1,c2,c3,c4 = st.columns(4)
    with c1: mcard(f"{metrics['accuracy']:.1%}",       "Accuracy",        "cyan")
    with c2: mcard(f"{metrics['roc_auc']:.3f}",        "ROC-AUC",         "green")
    with c3: mcard(f"{rpt['1']['recall']:.1%}",        "Fraud Recall",    "amber")
    with c4: mcard(f"{rpt['1']['precision']:.1%}",     "Fraud Precision", "cyan")

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        sec("Confusion Matrix")
        cpanel(plot_cm(metrics['confusion']))
    with c2:
        if model_choice=="Random Forest":
            sec("Feature Importance Ranking")
            cpanel(plot_fi(FEATURES, metrics['feature_importance']))
        else:
            sec("Precision-Recall Curve")
            cpanel(plot_pr(metrics))

    sec("Classification Report")
    rpt_df = pd.DataFrame({
        'Class':    ['Legitimate (0)','Fraudulent (1)','Macro Avg'],
        'Precision':[rpt['0']['precision'],rpt['1']['precision'],rpt['macro avg']['precision']],
        'Recall':   [rpt['0']['recall'],   rpt['1']['recall'],   rpt['macro avg']['recall']],
        'F1-Score': [rpt['0']['f1-score'], rpt['1']['f1-score'], rpt['macro avg']['f1-score']],
        'Support':  [int(rpt['0']['support']),int(rpt['1']['support']),
                     int(rpt['0']['support'])+int(rpt['1']['support'])],
    })
    st.dataframe(
        rpt_df.style.format({'Precision':'{:.4f}','Recall':'{:.4f}','F1-Score':'{:.4f}'})
                    .background_gradient(cmap='Blues', subset=['Precision','Recall','F1-Score'])
                    .set_properties(**{'text-align':'center'}),
        use_container_width=True, hide_index=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  📂  DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif "Dataset" in page:
    st.markdown('<div class="page-title"><span style="color:#00e5ff">📂</span> DATASET EXPLORER</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: mcard(f"{len(df):,}",           "Total Records",   "cyan",  "💾")
    with c2: mcard(f"{df['Class'].sum():,}",  "Fraud Cases",     "red",   "🚨")
    with c3: mcard(f"{len(df.columns)-1}",    "Features",        "green", "🔢")
    with c4: mcard(f"{df['Class'].mean():.2%}","Imbalance Ratio","amber", "⚖️")

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Data Browser")
    fc1, fc2 = st.columns([1,2])
    with fc1:
        n  = st.slider("Rows", 5, 100, 20)
        fc = st.selectbox("Filter", ["All","Legitimate (0)","Fraudulent (1)"])
    dsp = df.copy()
    if fc=="Legitimate (0)": dsp = df[df['Class']==0]
    if fc=="Fraudulent (1)": dsp = df[df['Class']==1]
    st.dataframe(
        dsp.head(n).style
           .map(lambda x:'background-color:#1a0010;color:#ff3366' if x==1 else '', subset=['Class'])
           .format({'Amount':'{:.2f}','V1':'{:.3f}','V2':'{:.3f}','V3':'{:.3f}',
                    'V4':'{:.3f}','V5':'{:.3f}','V6':'{:.3f}','V7':'{:.3f}','V8':'{:.3f}'}),
        use_container_width=True, hide_index=True
    )
    sec("Statistical Summary")
    st.dataframe(df.describe().round(4).style.background_gradient(cmap='Blues'),
                 use_container_width=True)
