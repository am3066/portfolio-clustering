import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import io
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================
# 0. CONFIGURATION GLOBALE
# ============================================
np.random.seed(42)
TOP_MARKET_CAP = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST', 'NFLX']

# ============================================
# 1. OPTIMISEUR DRIFT PSO
# ============================================
class DriftPSO:
    def __init__(self, n_particles=100, n_iterations=100, alpha=0.729, c1=1.49445, c2=1.49445, random_state=42):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state
        
    def optimize(self, fitness_func, n_dimensions, verbose=False):
        np.random.seed(self.random_state)
        positions = np.random.random((self.n_particles, n_dimensions))
        positions = positions / positions.sum(axis=1, keepdims=True)
        velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, n_dimensions))
        
        pbest_pos = positions.copy()
        pbest_score = np.array([fitness_func(p) for p in positions])
        
        gbest_pos = positions[np.argmax(pbest_score)]
        gbest_score = np.max(pbest_score)
        
        for iteration in range(self.n_iterations):
            mbest = np.mean(pbest_pos, axis=0)
            for i in range(self.n_particles):
                r1 = np.random.random(n_dimensions)
                r2 = np.random.random(n_dimensions)
                psi = np.random.randn(n_dimensions)
                
                cognitive = self.c1 * r1 * (pbest_pos[i] - positions[i])
                social = self.c2 * r2 * (gbest_pos - positions[i])
                drift = self.alpha * (mbest - positions[i]) * psi
                
                velocities[i] = drift + cognitive + social
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.maximum(positions[i], 0)
                
                if positions[i].sum() > 0:
                    positions[i] = positions[i] / positions[i].sum()
                else:
                    positions[i] = np.ones(n_dimensions) / n_dimensions
                
                fitness = fitness_func(positions[i])
                
                if fitness > pbest_score[i]:
                    pbest_score[i] = fitness
                    pbest_pos[i] = positions[i].copy()
                
                if fitness > gbest_score:
                    gbest_score = fitness
                    gbest_pos = positions[i].copy()
        
        return {'weights': gbest_pos, 'fitness': gbest_score}

# ============================================
# 2. CONFIGURATION UI
# ============================================
st.set_page_config(page_title="Pro Portfolio Optimizer", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle { text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 2rem; }
    .metric-card { background-color: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
    .metric-label { font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.8rem; color: #0f172a; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Pro Portfolio Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clustering K-Medoids & Optimisation DPSO - Pipeline Complet</div>', unsafe_allow_html=True)

# ============================================
# 3. PIPELINE DE TRAITEMENT (LOGIQUE EXACTE BATCH)
# ============================================
@st.cache_data
def run_full_pipeline(file, start_date, end_date, alpha):
    df = pd.read_csv(file)
    col_map = {c.lower(): c for c in df.columns}
    if 'date' in col_map: df.rename(columns={col_map['date']: 'date'}, inplace=True)
    if 'ticker' in col_map: df.rename(columns={col_map['ticker']: 'ticker'}, inplace=True)
    if 'close' in col_map: df.rename(columns={col_map['close']: 'close'}, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'])
    s_ts = pd.to_datetime(start_date)
    e_ts = pd.to_datetime(end_date)
    
    mask = (df['date'] >= s_ts) & (df['date'] <= e_ts)
    df = df[mask].copy()
    
    # Nettoyage
    cleaned_data = []
    for ticker in df['ticker'].unique():
        t_df = df[df['ticker'] == ticker].sort_values('date')
        date_range = pd.date_range(t_df['date'].min(), t_df['date'].max(), freq='D')
        full_df = pd.DataFrame({'date': date_range, 'ticker': ticker})
        merged = full_df.merge(t_df[['date', 'close']], on='date', how='left')
        merged['close'] = merged['close'].interpolate(method='linear').ffill().bfill()
        cleaned_data.append(merged)
    df_cleaned = pd.concat(cleaned_data, ignore_index=True).sort_values(['date', 'ticker'])
    
    # Lissage
    smoothed_data = []
    for ticker in df_cleaned['ticker'].unique():
        t_df = df_cleaned[df_cleaned['ticker'] == ticker].copy()
        t_df['close_smoothed'] = t_df['close'].ewm(alpha=alpha, adjust=False).mean()
        smoothed_data.append(t_df)
    df_smoothed = pd.concat(smoothed_data, ignore_index=True)
    
    # Rendements & Features
    df_smoothed['daily_return'] = df_smoothed.groupby('ticker')['close_smoothed'].pct_change()
    df_smoothed = df_smoothed.dropna(subset=['daily_return'])
    
    df_smoothed['year'] = df_smoothed['date'].dt.year
    df_smoothed['quarter'] = df_smoothed['date'].dt.quarter
    df_smoothed['year_quarter'] = df_smoothed['year'].astype(str) + '-Q' + df_smoothed['quarter'].astype(str)
    
    quarterly_feats = []
    for ticker in df_smoothed['ticker'].unique():
        t_df = df_smoothed[df_smoothed['ticker'] == ticker]
        for yq, group in t_df.groupby('year_quarter'):
            if len(group) < 5: continue
            quarterly_feats.append({
                'ticker': ticker,
                'year_quarter': yq,
                'quarter_start': group['date'].min(),
                'mean_return': group['daily_return'].mean(),
                'volatility': group['daily_return'].std()
            })
    df_features = pd.DataFrame(quarterly_feats)
    
    if df_features.empty: return pd.DataFrame(), pd.DataFrame(), None

    last_quarter_start = df_features['quarter_start'].max()
    df_clustering = df_features[df_features['quarter_start'] == last_quarter_start].copy()
    return df_clustering, df_smoothed, last_quarter_start

def determine_optimal_k(X_scaled, k_range=range(2, 11)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMedoids(n_clusters=k, random_state=42, init='k-medoids++')
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    optimal_k = k_range[np.argmax(silhouettes)]
    return optimal_k, {'k_range': list(k_range), 'inertias': inertias, 'silhouettes': silhouettes}

def perform_clustering_exact(df_features, n_clusters=None):
    X = df_features[['mean_return', 'volatility']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if n_clusters is None:
        n_clusters, elbow_data = determine_optimal_k(X_scaled)
    else:
        _, elbow_data = determine_optimal_k(X_scaled)
    
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, init='k-medoids++')
    df_features['cluster'] = kmedoids.fit_predict(X_scaled)
    medoid_indices = kmedoids.medoid_indices_
    medoid_tickers = df_features.iloc[medoid_indices]['ticker'].values
    silhouette = silhouette_score(X_scaled, df_features['cluster'])
    df_features[['mean_return_scaled', 'volatility_scaled']] = X_scaled
    return df_features, medoid_tickers, silhouette, scaler, X_scaled, n_clusters, elbow_data

def euclidean_distance_normalized(row, medoid_row):
    return np.sqrt((row['mean_return_scaled'] - medoid_row['mean_return_scaled'])**2 + 
                   (row['volatility_scaled'] - medoid_row['volatility_scaled'])**2)

def select_assets_exact(df_features, medoid_tickers, strategy_name, target_size):
    n_clusters = len(medoid_tickers)
    for k in range(n_clusters):
        med_row = df_features[df_features['ticker'] == medoid_tickers[k]].iloc[0]
        mask = df_features['cluster'] == k
        dists = df_features[mask].apply(lambda r: euclidean_distance_normalized(r, med_row), axis=1)
        df_features.loc[mask, 'distance_to_medoid'] = dists

    if "Stratégie 1" in strategy_name: return list(medoid_tickers)
    elif "Stratégie 2" in strategy_name:
        selected = set(medoid_tickers)
        per_cluster = (target_size - n_clusters) // n_clusters
        for k in range(n_clusters):
            cluster_df = df_features[(df_features['cluster'] == k) & (~df_features['ticker'].isin(medoid_tickers))]
            closest = cluster_df.nsmallest(per_cluster, 'distance_to_medoid')['ticker'].tolist()
            selected.update(closest)
        return list(selected)
    elif "Stratégie 3" in strategy_name:
        df_features['ratio'] = df_features['mean_return'] / (df_features['volatility'] + 1e-10)
        r_ret = df_features.sort_values('mean_return', ascending=False).head(target_size)['ticker'].tolist()
        r_vol = df_features.sort_values('volatility', ascending=True).head(target_size)['ticker'].tolist()
        r_rat = df_features.sort_values('ratio', ascending=False).head(target_size)['ticker'].tolist()
        r_dst = df_features.sort_values('distance_to_medoid', ascending=True).head(target_size)['ticker'].tolist()
        freq = {}
        for t in df_features['ticker']:
            freq[t] = sum([t in r_ret, t in r_vol, t in r_rat, t in r_dst])
        return [t for t, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:target_size]]
    else: 
        selected = set(medoid_tickers)
        top = [t for t in TOP_MARKET_CAP if t in df_features['ticker'].values]
        needed = target_size - len(selected)
        selected.update(top[:needed])
        return list(selected)

def calculate_sharpe_ratio(weights, returns, rf_rate=0):
    portfolio_returns = np.sum(weights * returns, axis=1)
    mean_return = np.mean(portfolio_returns) * 252
    volatility = np.std(portfolio_returns) * np.sqrt(252)
    return (mean_return - rf_rate) / volatility if volatility > 0 else -np.inf

def calculate_sortino_ratio(weights, returns, rf_rate=0):
    portfolio_returns = np.sum(weights * returns, axis=1)
    mean_return = np.mean(portfolio_returns) * 252
    neg = portfolio_returns[portfolio_returns < 0]
    dd = np.std(neg) * np.sqrt(252) if len(neg) > 0 else 0.0001
    return (mean_return - rf_rate) / dd

def calculate_adjusted_sharpe_ratio(weights, returns, rf_rate=0):
    sh = calculate_sharpe_ratio(weights, returns, rf_rate)
    if sh == -np.inf: return -np.inf
    p_ret = np.sum(weights * returns, axis=1)
    sk = stats.skew(p_ret)
    ku = stats.kurtosis(p_ret)
    return sh * (1 + (sk/6)*sh - (ku/24)*sh**2)

def optimize_portfolio(selected_assets, df_daily_returns, optimizer_type, pop_size=100, n_iter=100):
    returns_matrix = df_daily_returns[df_daily_returns['ticker'].isin(selected_assets)].pivot(index='date', columns='ticker', values='daily_return').fillna(0)
    for a in selected_assets:
        if a not in returns_matrix.columns: returns_matrix[a] = 0
    returns_matrix = returns_matrix[selected_assets]
    if returns_matrix.empty or len(returns_matrix) < 10: return None, None, None
    fitness_func = lambda w: calculate_sharpe_ratio(w, returns_matrix.values)
    
    if optimizer_type == "DPSO (Drift PSO)":
        optimizer = DriftPSO(n_particles=pop_size, n_iterations=n_iter, random_state=42)
        result = optimizer.optimize(fitness_func, len(selected_assets))
        weights = result['weights']
        score = result['fitness']
    else: 
        weights = np.ones(len(selected_assets)) / len(selected_assets)
        score = calculate_sharpe_ratio(weights, returns_matrix.values)
        
    p_ret = np.sum(weights * returns_matrix.values, axis=1)
    ann_ret = np.mean(p_ret) * 252
    ann_vol = np.std(p_ret) * np.sqrt(252)
    cum = (1 + p_ret).cumprod()
    max_dd = np.min(cum / np.maximum.accumulate(cum) - 1)
    calmar = abs(ann_ret / max_dd) if max_dd != 0 else 0
    
    metrics = {
        'sharpe_ratio': score, 'sortino_ratio': calculate_sortino_ratio(weights, returns_matrix.values),
        'adjusted_sharpe': calculate_adjusted_sharpe_ratio(weights, returns_matrix.values),
        'annual_return': ann_ret, 'annual_volatility': ann_vol, 'max_drawdown': max_dd, 'calmar_ratio': calmar
    }
    return weights, score, metrics

def generate_pdf_report(results_dict):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from io import BytesIO
        
        buff = BytesIO()
        doc = SimpleDocTemplate(buff, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#1E3A8A'), alignment=TA_CENTER, spaceAfter=20)
        h2_style = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#3B82F6'), spaceBefore=15, spaceAfter=10)
        normal = styles['Normal']
        
        story = [
            Paragraph("RAPPORT D'ANALYSE DE PORTEFEUILLE", title_style),
            Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y')}", ParagraphStyle('Date', parent=normal, alignment=TA_CENTER)),
            Spacer(1, 0.3*inch)
        ]
        
        # 1. CONFIGURATION
        story.append(Paragraph("1. Configuration de l'Analyse", h2_style))
        config_data = [
            ['Période', f"{results_dict['start_date']} au {results_dict['end_date']}"],
            ['Clustering', f"K-Medoids (K={results_dict['n_clusters']})"],
            ['Sélection', results_dict['selection_strategy']],
            ['Optimiseur', results_dict['optimizer']]
        ]
        t_conf = Table(config_data, colWidths=[2.5*inch, 3.5*inch])
        t_conf.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('GRID', (0,0), (-1,-1), 1, colors.white),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ]))
        story.append(t_conf)
        story.append(Spacer(1, 0.2*inch))

        # 2. PERFORMANCES
        story.append(Paragraph("2. Performances Clés", h2_style))
        perf_data = [
            ['Métrique', 'Valeur'],
            ['Sharpe Ratio', f"{results_dict['final_metrics']['sharpe_ratio']:.4f}"],
            ['Sortino Ratio', f"{results_dict['final_metrics']['sortino_ratio']:.4f}"],
            ['Calmar Ratio', f"{results_dict['final_metrics']['calmar_ratio']:.4f}"],
            ['Rendement Annuel', f"{results_dict['final_metrics']['annual_return']:.2%}"],
            ['Volatilité Annuelle', f"{results_dict['final_metrics']['annual_volatility']:.2%}"],
            ['Max Drawdown', f"{results_dict['final_metrics']['max_drawdown']:.2%}"]
        ]
        t_perf = Table(perf_data, colWidths=[2.5*inch, 3.5*inch])
        t_perf.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#DBEAFE')),
            ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#1E3A8A')),
            ('GRID', (0,0), (-1,-1), 1, colors.white)
        ]))
        story.append(t_perf)
        story.append(Spacer(1, 0.2*inch))

        # 3. ALLOCATION
        story.append(Paragraph("3. Allocation des Actifs", h2_style))
        weights_sorted = sorted(results_dict['weights'].items(), key=lambda x: x[1], reverse=True)
        alloc_data = [['Actif', 'Poids']] + [[k, f"{v:.2%}"] for k, v in weights_sorted]
        t_alloc = Table(alloc_data, colWidths=[3*inch, 3*inch])
        t_alloc.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('ALIGN', (1,0), (1,-1), 'RIGHT')
        ]))
        story.append(t_alloc)

        doc.build(story)
        buff.seek(0)
        return buff
    except Exception as e: return None

# ============================================
# 4. SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    uploaded_file = st.file_uploader("Fichier CSV (Ticker, Date, Close)", type=['csv'])
    
    with st.expander("📅 Période & Prétraitement", expanded=True):
        col1, col2 = st.columns(2)
        with col1: start_date = st.date_input("Début", value=pd.to_datetime("2015-01-01"))
        with col2: end_date = st.date_input("Fin (Train)", value=pd.to_datetime("2022-12-31"))
        alpha = st.slider("Alpha (EMA)", 0.001, 0.5, 0.01, 0.001)

    with st.expander("🎯 Clustering & Sélection", expanded=True):
        auto_k = st.checkbox("Déterminer K automatiquement (Elbow + Silhouette)", value=False)
        if not auto_k:
            n_clusters = st.slider("Nombre de Clusters (K)", 2, 8, 3)
        else:
            n_clusters = None
            
        strat = st.selectbox("Stratégie", ["Stratégie 1 : Médoïdes purs", "Stratégie 2 : Médoïdes + voisinage", "Stratégie 3 : Sélection multicritères (Recommandé)", "Stratégie 4 : Médoïdes + top ratio"], index=2)
        p_size = st.number_input("Taille Portefeuille", 5, 50, 15)

    with st.expander("🚀 Optimisation", expanded=True):
        optimizer_type = st.selectbox("Algorithme", ["DPSO (Drift PSO)", "Naïf (1/N)"], index=0)
        if optimizer_type == "DPSO (Drift PSO)":
            c1, c2 = st.columns(2)
            with c1: pso_pop = st.number_input("Population", 50, 500, 100, 50)
            with c2: pso_iter = st.number_input("Itérations", 50, 500, 100, 50)

    run_analysis = st.button("Lancer l'analyse", type="primary", use_container_width=True)

# ============================================
# 5. MAIN
# ============================================
if run_analysis and uploaded_file:
    try:
        progress_bar = st.progress(0)
        st.info("⚙️ Traitement des données en cours...")
        df_features, df_smoothed, selected_quarter_date = run_full_pipeline(uploaded_file, start_date, end_date, alpha)
        if df_features.empty: st.error("Erreur calcul features."); st.stop()
        progress_bar.progress(30)
        
        df_res, medoids, silhouette, scaler, X_scaled, final_k, elbow_data = perform_clustering_exact(df_features, n_clusters)
        progress_bar.progress(50)
        
        selection = select_assets_exact(df_res, medoids, strat, p_size)
        progress_bar.progress(70)
        
        if optimizer_type == "Naïf (1/N)":
            weights, score, metrics = optimize_portfolio(selection, df_smoothed, "Naïf", 100, 100)
        else:
            weights, score, metrics = optimize_portfolio(selection, df_smoothed, "DPSO (Drift PSO)", pso_pop, pso_iter)
        
        if weights is None: st.error("Erreur Optimisation"); st.stop()
        weights_dict = {a: w for a, w in zip(selection, weights)}
        _, _, naive_metrics = optimize_portfolio(selection, df_smoothed, "Naïf", 100, 100)
        
        progress_bar.progress(100)
        
        # ==================== CONTENU PRINCIPAL ====================
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Données", "🎯 Clustering", "💼 Portefeuille", "📈 Performances", "📄 Rapport"])
        
        # TAB 1 : DATA DASHBOARD PROFESSIONNEL
        with tab1:
            st.markdown("### 🔍 Vue d'ensemble du Dataset")
            
            # KPIs
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            def create_kpi(label, value):
                return f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
            
            kpi1.markdown(create_kpi("Observations Totales", f"{len(df_smoothed):,}"), unsafe_allow_html=True)
            kpi2.markdown(create_kpi("Jours de Trading", f"{df_smoothed['date'].nunique()}"), unsafe_allow_html=True)
            kpi3.markdown(create_kpi("Actifs Uniques", f"{df_smoothed['ticker'].nunique()}"), unsafe_allow_html=True)
            kpi4.markdown(create_kpi("Trimestre Clusterisé", selected_quarter_date.strftime('%Y-%m')), unsafe_allow_html=True)
            
            st.markdown("---")
            
            col_graph, col_desc = st.columns([1.5, 1])
            
            with col_graph:
                st.markdown("#### 📉 Distribution des Rendements Journaliers")
                fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
                df_smoothed['daily_return'].hist(bins=100, ax=ax_hist, color='#3B82F6', alpha=0.7)
                ax_hist.set_title("Histogramme des rendements")
                ax_hist.grid(False)
                st.pyplot(fig_hist)
                
                st.markdown("#### 🔥 Heatmap de Corrélation (Actifs Sélectionnés)")
                pivot = df_smoothed[df_smoothed['ticker'].isin(selection)].pivot(index='date', columns='ticker', values='daily_return')
                corr = pivot.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, ax=ax_corr, cmap="coolwarm", vmin=-1, vmax=1, annot=False)
                st.pyplot(fig_corr)

            with col_desc:
                st.markdown("#### 📋 Statistiques Descriptives")
                desc = df_smoothed[['close', 'close_smoothed', 'daily_return']].describe()
                st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)
                
                st.markdown("#### 📅 Couverture Temporelle")
                st.info(f"Du **{start_date.strftime('%d/%m/%Y')}** au **{end_date.strftime('%d/%m/%Y')}**")

        # TAB 2 : CLUSTERING
        with tab2:
            st.subheader(f"Partitionnement K-Medoids (K={final_k})")
            
            # Si K automatique, afficher les graphes Elbow + Silhouette
            if auto_k:
                st.markdown("### 📊 Méthodes de Sélection Automatique de K")
                col_elbow, col_sil = st.columns(2)
                
                with col_elbow:
                    st.markdown("#### Méthode du Coude (Elbow)")
                    fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
                    ax_elbow.plot(elbow_data['k_range'], elbow_data['inertias'], marker='o', color='#3B82F6')
                    ax_elbow.axvline(x=final_k, color='red', linestyle='--', label=f'K optimal = {final_k}')
                    ax_elbow.set_xlabel('Nombre de Clusters (K)')
                    ax_elbow.set_ylabel('Inertie')
                    ax_elbow.set_title('Méthode du Coude')
                    ax_elbow.legend()
                    ax_elbow.grid(True, alpha=0.3)
                    st.pyplot(fig_elbow)
                
                with col_sil:
                    st.markdown("#### Score de Silhouette")
                    fig_sil, ax_sil = plt.subplots(figsize=(6, 4))
                    ax_sil.plot(elbow_data['k_range'], elbow_data['silhouettes'], marker='s', color='#10B981')
                    ax_sil.axvline(x=final_k, color='red', linestyle='--', label=f'K optimal = {final_k}')
                    ax_sil.set_xlabel('Nombre de Clusters (K)')
                    ax_sil.set_ylabel('Score de Silhouette')
                    ax_sil.set_title('Analyse de Silhouette')
                    ax_sil.legend()
                    ax_sil.grid(True, alpha=0.3)
                    st.pyplot(fig_sil)
                
                st.success(f"✅ **K optimal déterminé automatiquement : {final_k}** (Score Silhouette : {silhouette:.3f})")
                st.markdown("---")
            
            # Calcul Effectifs
            counts = df_res['cluster'].value_counts().sort_index()
            
            c_graph, c_table = st.columns([2, 1])
            with c_graph:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors_list = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316']
                for k in range(final_k):
                    c_data = df_res[df_res['cluster'] == k]
                    ax.scatter(c_data['volatility'], c_data['mean_return'], c=colors_list[k%len(colors_list)], label=f'C{k} ({len(c_data)})', alpha=0.7)
                m_data = df_res[df_res['ticker'].isin(medoids)]
                ax.scatter(m_data['volatility'], m_data['mean_return'], c='black', marker='X', s=100, label='Médoïdes')
                ax.set_xlabel('Volatilité'); ax.set_ylabel('Rendement'); ax.legend()
                ax.grid(True, alpha=0.2)
                st.pyplot(fig)
            
            with c_table:
                st.markdown("##### 🔢 Répartition des Effectifs")
                fig_bar, ax_bar = plt.subplots(figsize=(4, 3))
                sns.barplot(x=counts.index, y=counts.values, ax=ax_bar, palette="viridis")
                ax_bar.set_xlabel("Cluster")
                ax_bar.set_ylabel("Nombre d'actifs")
                st.pyplot(fig_bar)
                
                count_df = pd.DataFrame({'Cluster': counts.index, 'Effectif': counts.values})
                st.dataframe(count_df, hide_index=True)

            st.markdown("##### 📋 Liste détaillée des actifs par Cluster")
            cols = st.columns(final_k)
            for k in range(final_k):
                members = df_res[df_res['cluster'] == k]['ticker'].tolist()
                with cols[k]:
                    st.success(f"**Cluster {k}** (Médoïde: {medoids[k]})")
                    st.text_area(f"Actifs C{k}", ", ".join(members), height=150)

        # TAB 3 : PORTEFEUILLE
        with tab3:
            st.subheader("Composition du Portefeuille")
            w_df = pd.DataFrame({'Actif': list(weights_dict.keys()), 'Poids': list(weights_dict.values())}).sort_values('Poids', ascending=False)
            
            c1, c2 = st.columns([2, 1])
            with c1:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x='Actif', y='Poids', data=w_df, palette='viridis', ax=ax)
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
            with c2:
                st.dataframe(w_df.style.format({'Poids': '{:.2%}'}), use_container_width=True)

        # TAB 4 : PERFORMANCES
        with tab4:
            st.header("Analyse Comparative")
            res_df = pd.DataFrame({
                'Métrique': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Rendement An.', 'Volatilité An.', 'Max Drawdown'],
                'Naïf (1/N)': [naive_metrics['sharpe_ratio'], naive_metrics['sortino_ratio'], naive_metrics['calmar_ratio'],
                               naive_metrics['annual_return'], naive_metrics['annual_volatility'], naive_metrics['max_drawdown']],
                'Optimisé (DPSO)': [metrics['sharpe_ratio'], metrics['sortino_ratio'], metrics['calmar_ratio'],
                             metrics['annual_return'], metrics['annual_volatility'], metrics['max_drawdown']]
            })
            st.dataframe(res_df.style.format("{:.4f}", subset=['Naïf (1/N)', 'Optimisé (DPSO)']), use_container_width=True)
            
            gain_s = (metrics['sharpe_ratio'] - naive_metrics['sharpe_ratio']) / abs(naive_metrics['sharpe_ratio'])
            st.metric("Gain Sharpe Ratio", f"{gain_s:+.2%}", delta_color="normal" if gain_s > 0 else "inverse")

        # TAB 5 : RAPPORT
        with tab5:
            s_gain = ((metrics['sharpe_ratio'] / naive_metrics['sharpe_ratio']) - 1) * 100
            r_gain = ((metrics['annual_return'] / naive_metrics['annual_return']) - 1) * 100
            v_gain = ((metrics['annual_volatility'] / naive_metrics['annual_volatility']) - 1) * 100
            
            if s_gain > 10: 
                conc_emoji = "✅"
                conc_txt = f"BONNE PERFORMANCE. L'optimisation {optimizer_type} a permis d'améliorer le Sharpe Ratio de {s_gain:.1f}%. Cette amélioration significative justifie l'utilisation de méthodes avancées."
            elif s_gain > 0: 
                conc_emoji = "⚠️"
                conc_txt = "AMÉLIORATION MODESTE. Le gain est marginal."
            else: 
                conc_emoji = "❌"
                conc_txt = "SOUS-PERFORMANCE. L'allocation naïve est préférable."

            st.markdown(f"""
            ### 📋 Synthèse de l'analyse
            **📅 Période analysée:** {start_date.strftime('%Y-%m-%d')} à {end_date.strftime('%Y-%m-%d')}  
            **🎯 Méthode de clustering:** K-medoids avec K={final_k} {'(automatique)' if auto_k else '(manuel)'} (Silhouette: {silhouette:.3f})  
            **📊 Stratégie de sélection:** {strat}  
            **🚀 Méthode d'optimisation:** {optimizer_type}  
            **💼 Nombre d'actifs dans le portefeuille:** {len(selection)}

            ### 🏆 Performances finales
            - **Sharpe Ratio:** {metrics['sharpe_ratio']:.4f}
            - **Sortino Ratio:** {metrics['sortino_ratio']:.4f}
            - **Adjusted Sharpe Ratio:** {metrics['adjusted_sharpe']:.4f}
            - **Rendement annualisé:** {metrics['annual_return']*100:.2f}%
            - **Volatilité annualisée:** {metrics['annual_volatility']*100:.2f}%
            - **Maximum Drawdown:** {metrics['max_drawdown']*100:.2f}%
            - **Calmar Ratio:** {metrics['calmar_ratio']:.4f}

            ### ✨ Amélioration vs Baseline naïf (1/N)
            - **Sharpe Ratio:** {s_gain:+.1f}% {"📈" if s_gain>0 else "📉"}
            - **Rendement:** {r_gain:+.1f}% {"📈" if r_gain>0 else "📉"}
            - **Volatilité:** {v_gain:+.1f}% {"📉" if v_gain>0 else "📈"}

            ### 🎯 Conclusion
            {conc_emoji} **{conc_txt.split('.')[0]}**
            
            {conc_txt}
            """)
            
            res_dict = {
                'start_date': start_date.strftime('%d/%m/%Y'), 
                'end_date': end_date.strftime('%d/%m/%Y'), 
                'n_clusters': final_k, 
                'selection_strategy': strat,
                'optimizer': optimizer_type, 
                'selected_assets': selection, 
                'weights': weights_dict, 
                'final_metrics': metrics,
                'naive_sharpe': naive_metrics['sharpe_ratio'], 
                'naive_return': naive_metrics['annual_return'],
                'calmar': metrics['calmar_ratio']
            }
            
            pdf = generate_pdf_report(res_dict)
            
            if pdf:
                st.download_button(
                    label="📄 Télécharger le Rapport Complet (PDF)",
                    data=pdf,
                    file_name="rapport_analyse_portefeuille.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            else:
                st.error("Erreur génération PDF. Vérifiez l'installation de reportlab.")

    except Exception as e:
        st.error(f"Une erreur est survenue : {str(e)}")
        st.exception(e)
elif not uploaded_file:
    st.info("👋 Veuillez charger votre fichier CSV (ex: nasdaq100_merged.csv) pour commencer.")