import sys
import os
sys.path.insert(0, 'src')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import shap
from preprocess import preprocess, SENSOR_COLS, DROP_SENSORS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FleetSense — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #1e2330;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] p {
    color: #94a3b8 !important;
    font-size: 0.8rem;
}

/* Top header bar */
.header-bar {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    margin: 0;
}
.header-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin: 2px 0 0 0;
    font-family: 'JetBrains Mono', monospace;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.kpi-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 10px 10px 0 0;
}
.kpi-card.healthy::before  { background: #22c55e; }
.kpi-card.warning::before  { background: #f59e0b; }
.kpi-card.critical::before { background: #ef4444; }
.kpi-card.total::before    { background: #3b82f6; }

.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748b;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.kpi-card.healthy  .kpi-value { color: #22c55e; }
.kpi-card.warning  .kpi-value { color: #f59e0b; }
.kpi-card.critical .kpi-value { color: #ef4444; }
.kpi-card.total    .kpi-value { color: #60a5fa; }
.kpi-sub {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #475569;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px;
    margin: 24px 0 14px 0;
}

/* Engine card in fleet grid */
.engine-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.eng-chip {
    width: 42px; height: 42px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.62rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.15s;
}
.eng-chip.healthy  { background: #052e16; color: #4ade80; border-color: #166534; }
.eng-chip.warning  { background: #431407; color: #fbbf24; border-color: #92400e; }
.eng-chip.critical { background: #450a0a; color: #f87171; border-color: #991b1b; }
.eng-chip.selected { outline: 2px solid #60a5fa; outline-offset: 2px; }

/* Metric chip */
.metric-row {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
}
.metric-chip {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 10px 16px;
    flex: 1;
    text-align: center;
}
.metric-chip .val {
    font-size: 1.3rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.metric-chip .lbl {
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge.healthy  { background: #052e16; color: #4ade80; }
.badge.warning  { background: #431407; color: #fbbf24; }
.badge.critical { background: #450a0a; color: #f87171; }

/* Plotly container */
.plot-container {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 4px;
    margin-bottom: 14px;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data & models ────────────────────────────────────────────────────────
@st.cache_data
def load_everything():
    X_train, y_train, X_test, y_test, feature_cols, train_df = preprocess(data_dir='data')
    lgb  = joblib.load('outputs/lgb_model.pkl')
    xgb  = joblib.load('outputs/xgb_model.pkl')
    rf   = joblib.load('outputs/rf_model.pkl')

    # Predict RUL for all 100 test engines
    preds = np.clip(lgb.predict(X_test), 0, 125)

    # Build fleet summary
    fleet = pd.DataFrame({
        'engine_id': range(1, 101),
        'predicted_rul': preds,
        'true_rul': y_test.values
    })
    fleet['status'] = fleet['predicted_rul'].apply(
        lambda r: 'critical' if r < 30 else ('warning' if r < 70 else 'healthy')
    )

    # SHAP for test set
    explainer   = shap.TreeExplainer(lgb)
    shap_values = explainer.shap_values(X_test)

    return lgb, xgb, rf, X_train, y_train, X_test, y_test, feature_cols, train_df, fleet, shap_values

with st.spinner("Initialising FleetSense..."):
    lgb, xgb, rf, X_train, y_train, X_test, y_test, feature_cols, train_df, fleet, shap_values = load_everything()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2rem;'>⚙️</div>
        <div style='font-size:1rem; font-weight:700; color:#f1f5f9;'>FleetSense</div>
        <div style='font-size:0.65rem; color:#475569; font-family: JetBrains Mono;'>NASA C-MAPSS · FD001</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Select Engine**")
    selected_engine = st.selectbox(
        "Engine ID", options=list(range(1, 101)),
        format_func=lambda x: f"Engine #{x:03d}  |  RUL: {fleet[fleet.engine_id==x].predicted_rul.values[0]:.0f} cycles",
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Filter Fleet View**")
    show_status = st.multiselect(
        "Status", ['healthy', 'warning', 'critical'],
        default=['healthy', 'warning', 'critical'],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Model Performance**")
    models_perf = {
        'LightGBM': {'MAE': 11.74, 'RMSE': 16.10, 'R²': 0.8385},
        'XGBoost':  {'MAE': 11.96, 'RMSE': 16.41, 'R²': 0.8323},
        'Random Forest': {'MAE': 12.75, 'RMSE': 16.94, 'R²': 0.8214},
    }
    for model_name, metrics in models_perf.items():
        icon = "🥇" if model_name == "LightGBM" else "  "
        st.markdown(f"""
        <div style='background:#0f172a; border:1px solid #1e293b; border-radius:8px;
                    padding:8px 12px; margin-bottom:8px;'>
            <div style='font-size:0.72rem; font-weight:600; color:#cbd5e1;'>{icon} {model_name}</div>
            <div style='font-size:0.65rem; color:#64748b; font-family:JetBrains Mono;'>
                MAE {metrics['MAE']} · R² {metrics['R²']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Vaidsys Technologies · ML Internship 2026")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <div style="font-size:2rem;">⚙️</div>
    <div>
        <p class="header-title">FleetSense — Predictive Maintenance Dashboard</p>
        <p class="header-sub">NASA C-MAPSS Turbofan · FD001 · LightGBM · RUL Regression · SHAP Explainability</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
n_total    = len(fleet)
n_healthy  = (fleet.status == 'healthy').sum()
n_warning  = (fleet.status == 'warning').sum()
n_critical = (fleet.status == 'critical').sum()

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card total">
        <div class="kpi-label">Total Engines</div>
        <div class="kpi-value">{n_total}</div>
        <div class="kpi-sub">FD001 test fleet</div>
    </div>
    <div class="kpi-card healthy">
        <div class="kpi-label">Healthy</div>
        <div class="kpi-value">{n_healthy}</div>
        <div class="kpi-sub">RUL ≥ 70 cycles</div>
    </div>
    <div class="kpi-card warning">
        <div class="kpi-label">Warning</div>
        <div class="kpi-value">{n_warning}</div>
        <div class="kpi-sub">30 ≤ RUL < 70 cycles</div>
    </div>
    <div class="kpi-card critical">
        <div class="kpi-label">Critical</div>
        <div class="kpi-value">{n_critical}</div>
        <div class="kpi-sub">RUL < 30 cycles</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Two column layout ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1.9], gap="medium")

# ── LEFT: Fleet Grid + Engine Detail ─────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-header">Fleet Overview</div>', unsafe_allow_html=True)

    filtered = fleet[fleet.status.isin(show_status)]
    chips_html = '<div class="engine-grid">'
    for _, row in fleet.iterrows():
        sel = 'selected' if row.engine_id == selected_engine else ''
        if row.status in show_status:
            chips_html += f'<div class="eng-chip {row.status} {sel}" title="Engine {row.engine_id} | RUL: {row.predicted_rul:.0f}">{row.engine_id:03d}</div>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)

    # ── Selected engine detail ────────────────────────────────────────────────
    eng_row = fleet[fleet.engine_id == selected_engine].iloc[0]
    pred_rul = eng_row.predicted_rul
    true_rul = eng_row.true_rul
    status   = eng_row.status

    st.markdown(f"""
    <div class="section-header" style="margin-top:24px;">
        Engine #{selected_engine:03d} Detail
        <span class="badge {status}" style="float:right; margin-top:-2px;">
            {'⚠ CRITICAL' if status=='critical' else ('⚡ WARNING' if status=='warning' else '✓ HEALTHY')}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-chip">
            <div class="val" style="color:{'#ef4444' if status=='critical' else '#f59e0b' if status=='warning' else '#22c55e'}">
                {pred_rul:.0f}
            </div>
            <div class="lbl">Predicted RUL</div>
        </div>
        <div class="metric-chip">
            <div class="val" style="color:#94a3b8">{true_rul:.0f}</div>
            <div class="lbl">True RUL</div>
        </div>
        <div class="metric-chip">
            <div class="val" style="color:#60a5fa">{abs(pred_rul - true_rul):.0f}</div>
            <div class="lbl">Error (cycles)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # RUL Gauge
    gauge_color = '#ef4444' if status=='critical' else '#f59e0b' if status=='warning' else '#22c55e'
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_rul,
        number={'font': {'size': 36, 'color': gauge_color, 'family': 'JetBrains Mono'},
                'suffix': ' cyc'},
        gauge={
            'axis': {'range': [0, 125], 'tickwidth': 1,
                     'tickcolor': '#334155', 'tickfont': {'color': '#64748b', 'size': 9}},
            'bar': {'color': gauge_color, 'thickness': 0.25},
            'bgcolor': '#0d0f14',
            'bordercolor': '#1e293b',
            'steps': [
                {'range': [0, 30],  'color': '#1a0505'},
                {'range': [30, 70], 'color': '#1a1005'},
                {'range': [70, 125],'color': '#051a0e'},
            ],
            'threshold': {
                'line': {'color': '#60a5fa', 'width': 2},
                'thickness': 0.75,
                'value': true_rul
            }
        }
    ))
    fig_gauge.update_layout(
        height=200, margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor='#111827', font_color='#e2e8f0'
    )
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Blue line = true RUL. Colored zones: red < 30, amber 30–70, green > 70.")

# ── RIGHT: Sensor trends + SHAP + Actual vs Predicted ────────────────────────
with col_right:

    # Sensor degradation chart
    st.markdown('<div class="section-header">Sensor Degradation — Engine #{:03d}</div>'.format(selected_engine),
                unsafe_allow_html=True)

    eng_data = train_df[train_df['unit_nr'] == selected_engine] if selected_engine <= train_df.unit_nr.max() else None

    active_sensors = [s for s in SENSOR_COLS if s not in DROP_SENSORS]
    top_sensors    = ['s_3', 's_4', 's_11', 's_9', 's_14', 's_15']

    if eng_data is not None and len(eng_data) > 0:
        fig_sensor = make_subplots(rows=2, cols=3, shared_xaxes=False,
                                   vertical_spacing=0.18, horizontal_spacing=0.1)
        positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
        colors    = ['#60a5fa','#34d399','#f472b6','#fb923c','#a78bfa','#facc15']

        for i, (sensor, pos, color) in enumerate(zip(top_sensors, positions, colors)):
            r, c = pos
            roll_col = f'{sensor}_roll_mean'
            fig_sensor.add_trace(go.Scatter(
                x=eng_data['time_cycles'], y=eng_data[sensor],
                mode='lines', line=dict(color=color, width=1, dash='dot'),
                opacity=0.4, name=sensor, showlegend=False
            ), row=r, col=c)
            if roll_col in eng_data.columns:
                fig_sensor.add_trace(go.Scatter(
                    x=eng_data['time_cycles'], y=eng_data[roll_col],
                    mode='lines', line=dict(color=color, width=2),
                    name=f'{sensor} trend', showlegend=False
                ), row=r, col=c)
            fig_sensor.update_xaxes(title_text='Cycle', title_font_size=9,
                                    gridcolor='#1e293b', color='#64748b', row=r, col=c)
            fig_sensor.update_yaxes(title_text=sensor, title_font_size=9,
                                    gridcolor='#1e293b', color='#64748b', row=r, col=c)

        fig_sensor.update_layout(
            height=300, paper_bgcolor='#111827', plot_bgcolor='#0d0f14',
            margin=dict(t=10, b=10, l=40, r=10), font_color='#94a3b8'
        )
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_sensor, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Dotted = raw sensor reading · Solid = 30-cycle rolling mean (trend)")
    else:
        st.info("Sensor trend data not available for this test engine in training history.")

    # Bottom row: SHAP + Actual vs Predicted
    col_shap, col_scatter = st.columns(2, gap="small")

    with col_shap:
        st.markdown('<div class="section-header">SHAP — Why this prediction?</div>',
                    unsafe_allow_html=True)

        eng_idx       = selected_engine - 1
        eng_shap      = shap_values[eng_idx]
        top_n         = 10
        top_idx       = np.argsort(np.abs(eng_shap))[::-1][:top_n]
        top_features  = [feature_cols[i] for i in top_idx]
        top_shap_vals = [eng_shap[i] for i in top_idx]
        colors_shap   = ['#ef4444' if v < 0 else '#22c55e' for v in top_shap_vals]

        fig_shap = go.Figure(go.Bar(
            x=top_shap_vals[::-1],
            y=top_features[::-1],
            orientation='h',
            marker_color=colors_shap[::-1],
            marker_line_width=0,
        ))
        fig_shap.update_layout(
            height=290, paper_bgcolor='#111827', plot_bgcolor='#0d0f14',
            margin=dict(t=10, b=10, l=10, r=10),
            font_color='#94a3b8', font_size=9,
            xaxis=dict(gridcolor='#1e293b', zeroline=True,
                       zerolinecolor='#334155', color='#64748b',
                       title='SHAP value', title_font_size=9),
            yaxis=dict(color='#94a3b8', tickfont_size=9)
        )
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("🟢 pushes RUL up (more life)  🔴 pushes RUL down (closer to failure)")

    with col_scatter:
        st.markdown('<div class="section-header">Actual vs Predicted RUL (All Engines)</div>',
                    unsafe_allow_html=True)

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=fleet.true_rul, y=fleet.predicted_rul,
            mode='markers',
            marker=dict(
                color=fleet.predicted_rul,
                colorscale=[[0,'#ef4444'],[0.3,'#f59e0b'],[0.7,'#22c55e'],[1,'#22c55e']],
                size=7, opacity=0.8, line=dict(width=0)
            ),
            text=[f"Engine {i}" for i in fleet.engine_id],
            hovertemplate='<b>%{text}</b><br>True: %{x:.0f}<br>Pred: %{y:.0f}<extra></extra>'
        ))
        # Perfect prediction line
        fig_scatter.add_trace(go.Scatter(
            x=[0, 125], y=[0, 125],
            mode='lines',
            line=dict(color='#334155', width=1, dash='dash'),
            showlegend=False
        ))
        # Highlight selected engine
        sel_true = fleet[fleet.engine_id == selected_engine].true_rul.values[0]
        sel_pred = fleet[fleet.engine_id == selected_engine].predicted_rul.values[0]
        fig_scatter.add_trace(go.Scatter(
            x=[sel_true], y=[sel_pred],
            mode='markers',
            marker=dict(color='#60a5fa', size=12, symbol='star',
                        line=dict(color='white', width=1)),
            name=f'Engine #{selected_engine}',
            showlegend=False,
            hovertemplate=f'<b>Engine #{selected_engine}</b><br>True: {sel_true:.0f}<br>Pred: {sel_pred:.0f}<extra></extra>'
        ))
        fig_scatter.update_layout(
            height=290, paper_bgcolor='#111827', plot_bgcolor='#0d0f14',
            margin=dict(t=10, b=10, l=40, r=10),
            font_color='#94a3b8', font_size=9,
            xaxis=dict(title='True RUL', title_font_size=9,
                       gridcolor='#1e293b', color='#64748b', range=[0,130]),
            yaxis=dict(title='Predicted RUL', title_font_size=9,
                       gridcolor='#1e293b', color='#64748b', range=[0,130]),
        )
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(f"⭐ = Engine #{selected_engine}. Closer to diagonal = better prediction.")
