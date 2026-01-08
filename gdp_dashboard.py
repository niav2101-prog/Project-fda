"""
Global GDP Explorer - Responsive Streamlit App
Loads: "Global GDP Explorer 2025 (World Bank  UN Data).csv"
Features:
- Responsive header and sidebar
- Filters: countries, metric, top-N
- Charts: top-N bar, scatter (GDP vs per capita), treemap (share), choropleth
- Table view and CSV download
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
from datetime import datetime
import difflib
import unicodedata
try:
    import pycountry
    PYCOUNTRY_AVAILABLE = True
except Exception:
    PYCOUNTRY_AVAILABLE = False

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Penjelajah PDB Global", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (modern & responsive)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    :root {
        --primary: #0066cc;
        --primary-light: #e6f0ff;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #1f2937;
        --light: #f9fafb;
        --border: #e5e7eb;
        --text: #374151;
    }
    
    * { 
        font-family: 'Poppins', sans-serif;
    }

    /* Main layout */
    body { background-color: #f3f4f6; }
    .main { padding: 0; }
    .stApp { background-color: #f3f4f6; }

    /* Header */
    .app-header { 
        padding: 40px 32px; 
        border-radius: 16px; 
        background: linear-gradient(135deg, #0066cc 0%, #0099ff 100%);
        margin-bottom: 30px; 
        box-shadow: 0 20px 40px rgba(0, 102, 204, 0.15);
        color: white;
    }
    .app-header h1 { 
        margin: 0; 
        color: white; 
        font-weight: 800; 
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .app-header p { 
        margin: 0; 
        color: rgba(255,255,255,0.9);
        font-weight: 500;
        font-size: 1.1rem;
    }
    .app-header .caption { 
        color: rgba(255,255,255,0.8);
        margin-top: 15px;
        font-size: 0.95rem;
    }

    /* Metrics Grid */
    .metrics-grid { 
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 30px 0;
    }
    .metric {
        padding: 20px;
        border-radius: 12px;
        background: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #0066cc;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .metric:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    .metric .label { 
        color: #6b7280; 
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .metric .value { 
        font-size: 1.6rem; 
        color: #0066cc; 
        font-weight: 800;
    }

    /* Sidebar */
    .sidebar-title {
        color: #0066cc;
        font-weight: 700;
        font-size: 1.2rem;
        margin: 20px 0 15px 0;
    }
    .sidebar-subtitle {
        color: #6b7280;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 15px 0 10px 0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #0066cc;
        border-bottom-color: #0066cc;
    }

    /* Cards */
    .card { 
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .card-title {
        color: #1f2937;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 16px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        background-color: #0066cc;
        color: white;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #0052a3;
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        padding: 10px 12px;
        font-size: 0.95rem;
        transition: border-color 0.3s;
    }
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #0066cc;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .app-header {
            padding: 24px 16px;
        }
        .app-header h1 {
            font-size: 1.8rem;
        }
        .metrics-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Helper: parse and clean CSV values similar to project loader but simplified
@st.cache_data(ttl=86400)
def load_gdp(path):
    if not os.path.exists(path):
        st.error(f"Dataset tidak ditemukan: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, sep=",", encoding='utf-8', on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    # Rename columns to consistent names
    df = df.rename(columns={
        'Country': 'country',
        'GDP (nominal, 2023)': 'gdp_nominal',
        'GDP (abbrev.)': 'gdp_abbrev',
        'GDP Growth': 'gdp_growth',
        'Population 2023': 'population_2023',
        'GDP per capita': 'gdp_per_capita',
        'Share of World GDP': 'share_world'
    })

    def parse_money(x):
        if pd.isna(x):
            return np.nan
        s = str(x).replace('$','').replace(',','').replace(' ','').replace('−','-')
        if 'trillion' in s.lower():
            return float(''.join(ch for ch in s if ch.isdigit() or ch in '.-')) * 1e12
        if 'billion' in s.lower():
            return float(''.join(ch for ch in s if ch.isdigit() or ch in '.-')) * 1e9
        try:
            return float(''.join(ch for ch in s if ch.isdigit() or ch in '.-'))
        except:
            return np.nan

    df['gdp_nominal_numeric'] = df['gdp_nominal'].apply(parse_money)
    df['gdp_per_capita_numeric'] = df['gdp_per_capita'].astype(str).str.replace('[\$,]','', regex=True).astype(str).apply(lambda s: float(s.replace(',','')) if s.replace(',','').replace('.','',1).lstrip('-').isdigit() else np.nan)
    df['gdp_growth_numeric'] = df['gdp_growth'].astype(str).str.replace('%','').str.replace('−','-').astype(str).apply(lambda s: float(s) if s.replace('.','',1).lstrip('-').isdigit() else np.nan)
    df['population_2023'] = df['population_2023'].astype(str).str.replace(',','').apply(lambda s: float(s) if s.replace('.','',1).isdigit() else np.nan)
    df['share_world_numeric'] = df['share_world'].astype(str).str.replace('%','').apply(lambda s: float(s) if s.replace('.','',1).isdigit() else np.nan)
    return df

# ===== MACHINE LEARNING MODELS =====

# 1. Random Forest: Prediksi GDP per Kapita
@st.cache_data(ttl=86400)
def train_rf_gdp_per_capita(df):
    """Random Forest untuk prediksi GDP per Kapita"""
    model_data = df[['gdp_nominal_numeric', 'population_2023', 'gdp_growth_numeric', 'gdp_per_capita_numeric']].dropna()
    if len(model_data) < 5:
        return None, None, None, None
    
    X = model_data[['gdp_nominal_numeric', 'population_2023', 'gdp_growth_numeric']]
    y = model_data['gdp_per_capita_numeric']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, r2, rmse, (X, y)

# 2. SVM: Klasifikasi Kategori Income
@st.cache_data(ttl=86400)
def train_svm_income_classification(df):
    """SVM untuk klasifikasi negara (High/Medium/Low Income)"""
    model_data = df[['gdp_per_capita_numeric', 'gdp_growth_numeric', 'population_2023']].dropna()
    if len(model_data) < 5:
        return None, None, None, None
    
    # Tentukan threshold untuk kategori income
    q75 = model_data['gdp_per_capita_numeric'].quantile(0.75)
    q25 = model_data['gdp_per_capita_numeric'].quantile(0.25)
    
    categories = []
    for val in model_data['gdp_per_capita_numeric']:
        if val >= q75:
            categories.append(2)  # High
        elif val >= q25:
            categories.append(1)  # Medium
        else:
            categories.append(0)  # Low
    
    X = model_data[['gdp_growth_numeric', 'population_2023']]
    y = np.array(categories)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='rbf', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, scaler, (X_scaled, y)

# 3. KMeans: Clustering Negara
@st.cache_data(ttl=86400)
def train_kmeans_clustering(df, n_clusters=4):
    """KMeans clustering untuk pengelompokan negara"""
    model_data = df[['gdp_nominal_numeric', 'gdp_per_capita_numeric', 'gdp_growth_numeric', 'population_2023']].dropna()
    if len(model_data) < 5:
        return None, None, None, None
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_data)
    
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, clusters)
    
    return model, clusters, scaler, silhouette

# 4. Random Forest: Prediksi Pertumbuhan GDP
@st.cache_data(ttl=86400)
def train_rf_gdp_growth(df):
    """Random Forest untuk prediksi pertumbuhan GDP"""
    model_data = df[['gdp_nominal_numeric', 'gdp_per_capita_numeric', 'population_2023', 'gdp_growth_numeric']].dropna()
    if len(model_data) < 5:
        return None, None, None, None
    
    X = model_data[['gdp_nominal_numeric', 'gdp_per_capita_numeric', 'population_2023']]
    y = model_data['gdp_growth_numeric']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, r2, rmse, (X, y)

# Load dataset
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, "data", "gdp_2025.csv")


with st.spinner('⏳ Memuat dataset GDP...'):
    gdf = load_gdp(DATA_PATH)

if gdf.empty:
    st.stop()

# Header
st.markdown("""
<div class='app-header'>
    <h1>🌍 Penjelajah PDB Global</h1>
    <p>Analisis data ekonomi dunia secara interaktif</p>
    <div class='caption'>Sumber data: World Bank / UN Data — versi dataset: 'Global GDP Explorer 2025'</div>
</div>
""", unsafe_allow_html=True)
# Pemetaan negara (ISO + fuzzy fallback)
unique_countries = sorted(gdf['country'].dropna().unique().tolist())

def normalize_name(s):
    if pd.isna(s):
        return ''
    s = unicodedata.normalize('NFKD', str(s))
    s = ''.join(ch for ch in s if ch.isalnum() or ch.isspace())
    return s.strip().lower()

# Build pycountry lookup map (if available)
pycountry_map = {}
if PYCOUNTRY_AVAILABLE:
    for c in pycountry.countries:
        pycountry_map[normalize_name(getattr(c, 'name', ''))] = c.alpha_3
        if hasattr(c, 'official_name'):
            pycountry_map[normalize_name(getattr(c, 'official_name'))] = c.alpha_3

# Map dataset countries to ISO-3 where possible
def map_country_iso(name, cutoff=0.85):
    n = normalize_name(name)
    if not n:
        return None
    if n in pycountry_map:
        return pycountry_map[n]
    # try pycountry.lookup direct (handles many aliases)
    if PYCOUNTRY_AVAILABLE:
        try:
            res = pycountry.countries.lookup(name)
            return res.alpha_3
        except Exception:
            pass
    # fuzzy match against pycountry names
    if pycountry_map:
        m = difflib.get_close_matches(n, list(pycountry_map.keys()), n=1, cutoff=cutoff)
        if m:
            return pycountry_map[m[0]]
    # fallback: no mapping
    return None

initial_mapping = {c: map_country_iso(c) for c in unique_countries}
mapping_df = pd.DataFrame({'country': list(initial_mapping.keys()), 'iso_alpha3': list(initial_mapping.values())})
mapped_count = mapping_df['iso_alpha3'].notna().sum()
unmapped = mapping_df[mapping_df['iso_alpha3'].isna()]['country'].tolist()

# Show mapping summary in sidebar
with st.sidebar.expander("📊 Informasi Dataset", expanded=True):
    st.metric("Total Negara", len(mapping_df))
    st.metric("Negara Terpetakan", f"{mapped_count}/{len(mapping_df)}")
    
    if not PYCOUNTRY_AVAILABLE:
        st.info("💡 **Tip:** Pasang `pycountry` untuk hasil yang lebih akurat: `pip install pycountry`")



# apply initial mapping to dataframe
if 'iso_alpha3' not in gdf.columns:
    gdf['iso_alpha3'] = gdf['country'].map(initial_mapping)

# ===== SIDEBAR FILTERS =====
st.sidebar.markdown("---")
st.sidebar.title("⚙️ Filter & Pengaturan")

# Pemetaan metrik
metric_options = {
    "PDB (nominal)": "gdp_nominal_numeric",
    "PDB per kapita": "gdp_per_capita_numeric",
    "Pertumbuhan PDB": "gdp_growth_numeric"
}

metric_display = st.sidebar.selectbox("📈 Pilih metrik analisis:", options=list(metric_options.keys()))
metric_col = metric_options[metric_display]

# Negara
all_countries = sorted(gdf['country'].dropna().unique().tolist())
st.sidebar.markdown("<p class='sidebar-subtitle'>🌐 Pilihan Negara</p>", unsafe_allow_html=True)

search_country = st.sidebar.text_input("Cari negara:", placeholder="Ketik nama negara...", help="Pencarian otomatis akan memfilter daftar")

if search_country:
    country_options = [c for c in all_countries if search_country.lower() in c.lower()]
    if not country_options:
        st.sidebar.info(f"❌ Tidak ada negara yang cocok dengan '{search_country}'")
        country_options = all_countries
else:
    country_options = all_countries

default_selection = [c for c in all_countries[:10] if c in country_options]
if not default_selection:
    default_selection = country_options[:5] if country_options else []

selected_countries = st.sidebar.multiselect("Pilih negara:", options=country_options, default=default_selection, help="Kosongkan untuk menampilkan semua negara")

# Pengaturan chart
st.sidebar.markdown("<p class='sidebar-subtitle'>📊 Pengaturan Chart</p>", unsafe_allow_html=True)
top_n = st.sidebar.slider("Negara teratas (Top N):", min_value=5, max_value=50, value=15, step=5)
show_map = st.sidebar.checkbox("Tampilkan peta choropleth", value=True, help="Peta visualisasi PDB global")
use_iso_map = st.sidebar.checkbox("Gunakan kode ISO untuk peta", value=True, help="Hasil lebih akurat dengan ISO-3")

# Tabel
st.sidebar.markdown("<p class='sidebar-subtitle'>📋 Pengaturan Tabel</p>", unsafe_allow_html=True)
available_cols = ['country','gdp_nominal','gdp_nominal_numeric','gdp_per_capita','gdp_per_capita_numeric','population_2023','gdp_growth','share_world']
display_cols = st.sidebar.multiselect("Kolom yang ditampilkan:", options=available_cols, default=['country','gdp_nominal_numeric','gdp_per_capita_numeric','gdp_growth','population_2023'], help="Pilih kolom untuk ditampilkan di tabel")
rows = st.sidebar.number_input("Baris per halaman:", min_value=10, max_value=500, value=50, step=10)

# Main layout
col1, col2 = st.columns([3, 2])
with col1:
    # Top N bar
    top_df = gdf.sort_values(metric_col, ascending=False).head(top_n)
    st.markdown(f"<div class='card'><div class='card-title'>🥇 Top {top_n} - {metric_display}</div></div>", unsafe_allow_html=True)
    fig = px.bar(top_df, x=metric_col, y='country', orientation='h', color=metric_col, labels={metric_col:metric_display, 'country':'Negara'}, color_continuous_scale='Blues')
    fig.update_layout(height=420, plot_bgcolor='rgba(0,0,0,0)', xaxis={'gridcolor':'#e5e7eb'}, yaxis={'automargin':True}, showlegend=False)
    fig.update_traces(hovertemplate='<b>%{y}</b><br>%{x:,.0f}<extra></extra>')
    st.plotly_chart(fig, use_container_width=True)

    # Scatter
    st.markdown("<div class='card'><div class='card-title'>📊 Hubungan PDB vs PDB per Kapita</div></div>", unsafe_allow_html=True)
    fig2 = px.scatter(gdf, x='gdp_per_capita_numeric', y='gdp_nominal_numeric', size='population_2023', hover_name='country', labels={'gdp_per_capita_numeric':'PDB per kapita (USD)','gdp_nominal_numeric':'PDB Nominal (USD)'}, size_max=60, color='gdp_nominal_numeric', color_continuous_scale='Viridis')
    fig2.update_layout(height=420, plot_bgcolor='rgba(0,0,0,0)', xaxis={'gridcolor':'#e5e7eb'}, yaxis={'gridcolor':'#e5e7eb'})
    fig2.update_traces(marker=dict(opacity=0.7), selector=dict(mode='markers'), hovertemplate='<b>%{hovertext}</b><br>PDB per kapita: %{x:,.0f}<br>PDB: %{y:,.0f}<extra></extra>')
    st.plotly_chart(fig2, use_container_width=True)

    # Treemap share
    st.markdown("<div class='card'><div class='card-title'>🌐 Porsi PDB Dunia (Treemap)</div></div>", unsafe_allow_html=True)
    treedf = gdf.dropna(subset=['share_world_numeric'])
    if not treedf.empty:
        fig3 = px.treemap(treedf.head(30), path=['country'], values='share_world_numeric', color='share_world_numeric', color_continuous_scale='Oranges', labels={'share_world_numeric':'Persentase (%)'})
        fig3.update_layout(height=420)
        fig3.update_traces(hovertemplate='<b>%{label}</b><br>Persentase: %{value:.2f}%<extra></extra>')
        st.plotly_chart(fig3, use_container_width=True)

with col2:
    # Cards (ringkasan)
    total_gdp = gdf['gdp_nominal_numeric'].sum(skipna=True)
    avg_per_capita = gdf['gdp_per_capita_numeric'].mean(skipna=True)
    total_countries = gdf['country'].nunique()
    avg_growth = gdf['gdp_growth_numeric'].mean(skipna=True)

    st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class='metric'>
        <div class='label'>📍 Total Negara</div>
        <div class='value'>{total_countries}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='metric'>
        <div class='label'>💰 Total PDB Dunia</div>
        <div class='value'>${total_gdp/1e12:.1f}T</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='metric'>
        <div class='label'>👨 Rata-rata PDB per Kapita</div>
        <div class='value'>${avg_per_capita:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='metric'>
        <div class='label'>📈 Rata-rata Pertumbuhan</div>
        <div class='value'>{avg_growth:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if show_map:
        st.markdown("<div class='card'><div class='card-title'>🗺️ Peta Choropleth</div></div>", unsafe_allow_html=True)
        map_df = gdf.dropna(subset=['gdp_nominal_numeric'])
        
        if use_iso_map:
            map_df = map_df.dropna(subset=['iso_alpha3'])
            locations = 'iso_alpha3'
            locationmode = 'ISO-3'
        else:
            locations = 'country'
            locationmode = 'country names'
        
        try:
            figmap = px.choropleth(map_df, locations=locations, locationmode=locationmode, color='gdp_nominal_numeric', hover_name='country', color_continuous_scale='Blues', labels={'gdp_nominal_numeric':'PDB (USD)'})
            figmap.update_layout(height=480, margin={'r':0,'t':0,'l':0,'b':0}, geo=dict(bgcolor='rgba(0,0,0,0)'))
            figmap.update_traces(hovertemplate='<b>%{hovertext}</b><br>PDB: %{z:,.0f}<extra></extra>')
            st.plotly_chart(figmap, use_container_width=True)
        except Exception as e:
            st.error('⚠️ Gagal membuat peta. Coba gunakan kode ISO.')
        
        if use_iso_map:
            missing_on_map = sorted(list(set(gdf['country'].dropna()) - set(map_df['country'].dropna())))
            if missing_on_map:
                with st.expander(f"ℹ️ {len(missing_on_map)} negara tidak ditampilkan di peta"):
                    st.write(', '.join(missing_on_map))

# ===== MACHINE LEARNING SECTION =====
st.markdown('---')
st.markdown("""
<div class='card'><div class='card-title'>🤖 Analisis Machine Learning</div></div>
""", unsafe_allow_html=True)

ml_tabs = st.tabs(["🌳 Random Forest: GDP/Kapita", "🎯 SVM: Klasifikasi Income", "📊 Clustering Negara", "📈 RF: Prediksi Pertumbuhan"])

# TAB 1: Random Forest - GDP per Kapita
with ml_tabs[0]:
    st.subheader("🌳 Prediksi GDP per Kapita dengan Random Forest")
    
    with st.spinner("🔄 Training model Random Forest..."):
        rf_model, rf_r2, rf_rmse, rf_data = train_rf_gdp_per_capita(gdf)
    
    if rf_model:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{rf_r2:.3f}", help="Seberapa baik model menjelaskan variansi data (0-1)")
        with col2:
            st.metric("RMSE", f"${rf_rmse:,.0f}", help="Root Mean Squared Error - rata-rata error prediksi")
        with col3:
            st.metric("Akurasi", f"{rf_r2*100:.1f}%", help="Persentase variansi yang dijelaskan")
        
        # Feature importance
        X, y = rf_data
        feature_importance = pd.DataFrame({
            'Fitur': ['GDP Nominal', 'Populasi', 'Pertumbuhan GDP'],
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(feature_importance, x='Importance', y='Fitur', orientation='h', color='Importance', color_continuous_scale='Blues')
        fig_importance.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Prediksi untuk negara yang dipilih
        st.subheader("🔮 Prediksi untuk Negara Tertentu")
        pred_countries = st.multiselect("Pilih negara untuk prediksi:", options=gdf['country'].dropna().unique(), default=gdf['country'].dropna().unique()[:5], key="rf_gdp_countries")
        
        if pred_countries:
            pred_df = gdf[gdf['country'].isin(pred_countries)][['country', 'gdp_nominal_numeric', 'population_2023', 'gdp_growth_numeric', 'gdp_per_capita_numeric']].dropna()
            if not pred_df.empty:
                X_pred = pred_df[['gdp_nominal_numeric', 'population_2023', 'gdp_growth_numeric']]
                predictions = rf_model.predict(X_pred)
                
                pred_result = pd.DataFrame({
                    'Negara': pred_df['country'].values,
                    'Aktual GDP/Kapita': pred_df['gdp_per_capita_numeric'].apply(lambda x: f"${x:,.0f}"),
                    'Prediksi RF': [f"${x:,.0f}" for x in predictions],
                    'Error': [f"${abs(a-p):,.0f}" for a, p in zip(pred_df['gdp_per_capita_numeric'].values, predictions)]
                })
                st.dataframe(pred_result, use_container_width=True)
    else:
        st.warning("⚠️ Data tidak cukup untuk melatih model")

# TAB 2: SVM - Klasifikasi Income
with ml_tabs[1]:
    st.subheader("🎯 Klasifikasi Kategori Income dengan SVM")
    
    with st.spinner("🔄 Training model SVM..."):
        svm_model, svm_accuracy, svm_scaler, svm_data = train_svm_income_classification(gdf)
    
    if svm_model:
        st.metric("Akurasi Klasifikasi", f"{svm_accuracy*100:.1f}%", help="Persentase prediksi yang benar")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🟢 **High Income**: GDP/kapita tinggi (Top 25%)")
        with col2:
            st.info("🟡 **Medium Income**: GDP/kapita menengah (25%-75%)")
        with col3:
            st.info("🔴 **Low Income**: GDP/kapita rendah (Bottom 25%)")
        
        # Klasifikasi semua negara
        X_scaled, y = svm_data
        model_data = gdf[['gdp_per_capita_numeric', 'gdp_growth_numeric', 'population_2023']].dropna()
        
        # Tentukan threshold untuk kategori
        q75 = model_data['gdp_per_capita_numeric'].quantile(0.75)
        q25 = model_data['gdp_per_capita_numeric'].quantile(0.25)
        
        predictions_all = svm_model.predict(X_scaled)
        
        income_map = {0: '🔴 Low Income', 1: '🟡 Medium Income', 2: '🟢 High Income'}
        model_data_copy = model_data.reset_index(drop=True)
        model_data_copy['Kategori'] = [income_map[p] for p in predictions_all]
        
        # Ambil dari gdf original
        classified = gdf[['country', 'gdp_per_capita_numeric']].dropna()
        classified_indices = model_data.index
        classified['Kategori'] = [income_map[p] for p in predictions_all]
        
        st.subheader("📊 Hasil Klasifikasi Negara")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            high = (np.array(predictions_all) == 2).sum()
            st.metric("🟢 High Income", high)
        with col2:
            medium = (np.array(predictions_all) == 1).sum()
            st.metric("🟡 Medium Income", medium)
        with col3:
            low = (np.array(predictions_all) == 0).sum()
            st.metric("🔴 Low Income", low)
        
        # Pie chart kategori
        category_counts = [low, medium, high]
        fig_pie = px.pie(values=category_counts, names=['Low Income', 'Medium Income', 'High Income'], 
                        color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981'])
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tabel klasifikasi
        st.subheader("📋 Daftar Negara per Kategori")
        filter_income = st.selectbox("Filter kategori:", options=['Semua', 'High Income', 'Medium Income', 'Low Income'])
        
        # Rebuild classification result properly
        model_countries = gdf[['country', 'gdp_per_capita_numeric', 'gdp_growth_numeric', 'population_2023']].dropna()
        X_for_pred = model_countries[['gdp_growth_numeric', 'population_2023']]
        X_scaled_pred = svm_scaler.transform(X_for_pred)
        pred_categories = svm_model.predict(X_scaled_pred)
        
        model_countries['Kategori'] = [income_map[p] for p in pred_categories]
        
        if filter_income != 'Semua':
            display_countries = model_countries[model_countries['Kategori'].str.contains(filter_income)]
        else:
            display_countries = model_countries
        
        display_countries_show = display_countries[['country', 'Kategori', 'gdp_per_capita_numeric']].sort_values('gdp_per_capita_numeric', ascending=False)
        display_countries_show.columns = ['Negara', 'Kategori', 'GDP per Kapita']
        st.dataframe(display_countries_show, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Data tidak cukup untuk melatih model SVM")

# TAB 3: KMeans Clustering
with ml_tabs[2]:
    st.subheader("📊 Clustering Negara dengan KMeans")
    
    n_clusters_slider = st.slider("Jumlah kluster:", min_value=2, max_value=8, value=4)
    
    with st.spinner(f"🔄 Training KMeans dengan {n_clusters_slider} kluster..."):
        kmeans_model, cluster_labels, kmeans_scaler, silhouette = train_kmeans_clustering(gdf, n_clusters=n_clusters_slider)
    
    if kmeans_model:
        st.metric("Silhouette Score", f"{silhouette:.3f}", help="Seberapa baik pengelompokan (nilai lebih tinggi lebih baik, range: -1 hingga 1)")
        
        # Tambahkan cluster labels ke dataframe
        cluster_df = gdf[['country', 'gdp_nominal_numeric', 'gdp_per_capita_numeric', 'gdp_growth_numeric', 'population_2023']].dropna()
        cluster_df['Cluster'] = cluster_labels
        
        # Visualisasi scatter dengan clustering
        fig_cluster = px.scatter(cluster_df, x='gdp_per_capita_numeric', y='gdp_nominal_numeric', 
                                size='population_2023', color='Cluster', hover_name='country',
                                labels={'gdp_per_capita_numeric': 'GDP per Kapita (USD)', 
                                       'gdp_nominal_numeric': 'GDP Nominal (USD)'},
                                color_continuous_scale='Viridis')
        fig_cluster.update_layout(height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Statistik cluster
        st.subheader("📈 Statistik Cluster")
        
        for i in range(n_clusters_slider):
            cluster_countries = cluster_df[cluster_df['Cluster'] == i]
            with st.expander(f"Cluster {i} ({len(cluster_countries)} negara)"):
                avg_gdp = cluster_countries['gdp_nominal_numeric'].mean()
                avg_per_capita = cluster_countries['gdp_per_capita_numeric'].mean()
                avg_growth = cluster_countries['gdp_growth_numeric'].mean()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rata-rata GDP", f"${avg_gdp/1e12:.2f}T")
                with col2:
                    st.metric("Rata-rata GDP/Kapita", f"${avg_per_capita:,.0f}")
                with col3:
                    st.metric("Rata-rata Pertumbuhan", f"{avg_growth:.2f}%")
                
                st.write("**Negara dalam cluster ini:**")
                cluster_list = cluster_countries[['country', 'gdp_nominal_numeric', 'gdp_per_capita_numeric']].sort_values('gdp_nominal_numeric', ascending=False)
                cluster_list.columns = ['Negara', 'GDP Nominal', 'GDP per Kapita']
                st.dataframe(cluster_list, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Data tidak cukup untuk clustering")

# TAB 4: Random Forest - Prediksi Pertumbuhan GDP
with ml_tabs[3]:
    st.subheader("📈 Prediksi Pertumbuhan GDP dengan Random Forest")
    
    with st.spinner("🔄 Training model Random Forest (Pertumbuhan)..."):
        rf_growth_model, rf_growth_r2, rf_growth_rmse, rf_growth_data = train_rf_gdp_growth(gdf)
    
    if rf_growth_model:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{rf_growth_r2:.3f}", help="Seberapa baik model menjelaskan variansi pertumbuhan")
        with col2:
            st.metric("RMSE", f"{rf_growth_rmse:.3f}%", help="Root Mean Squared Error prediksi pertumbuhan")
        with col3:
            st.metric("Akurasi", f"{rf_growth_r2*100:.1f}%", help="Persentase variansi yang dijelaskan")
        
        # Feature importance
        X_growth, y_growth = rf_growth_data
        feature_importance_growth = pd.DataFrame({
            'Fitur': ['GDP Nominal', 'GDP per Kapita', 'Populasi'],
            'Importance': rf_growth_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_importance_growth = px.bar(feature_importance_growth, x='Importance', y='Fitur', orientation='h', color='Importance', color_continuous_scale='Greens')
        fig_importance_growth.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_importance_growth, use_container_width=True)
        
        # Prediksi untuk negara yang dipilih
        st.subheader("🔮 Prediksi Pertumbuhan untuk Negara Tertentu")
        pred_countries_growth = st.multiselect("Pilih negara untuk prediksi pertumbuhan:", 
                                              options=gdf['country'].dropna().unique(), 
                                              default=gdf['country'].dropna().unique()[:5],
                                              key="rf_growth_countries")
        
        if pred_countries_growth:
            pred_df_growth = gdf[gdf['country'].isin(pred_countries_growth)][['country', 'gdp_nominal_numeric', 'gdp_per_capita_numeric', 'population_2023', 'gdp_growth_numeric']].dropna()
            if not pred_df_growth.empty:
                X_pred_growth = pred_df_growth[['gdp_nominal_numeric', 'gdp_per_capita_numeric', 'population_2023']]
                predictions_growth = rf_growth_model.predict(X_pred_growth)
                
                pred_result_growth = pd.DataFrame({
                    'Negara': pred_df_growth['country'].values,
                    'Aktual Pertumbuhan': pred_df_growth['gdp_growth_numeric'].apply(lambda x: f"{x:.2f}%"),
                    'Prediksi RF': [f"{x:.2f}%" for x in predictions_growth],
                    'Error': [f"{abs(a-p):.2f}%" for a, p in zip(pred_df_growth['gdp_growth_numeric'].values, predictions_growth)]
                })
                st.dataframe(pred_result_growth, use_container_width=True)
    else:
        st.warning("⚠️ Data tidak cukup untuk melatih model")

# Tabel data
st.markdown('---')
st.markdown("<div class='card'><div class='card-title'>📋 Data Lengkap</div></div>", unsafe_allow_html=True)

filter_df = gdf.copy()
if selected_countries:
    filter_df = filter_df[filter_df['country'].isin(selected_countries)]

# Apply search if used
if search_country and not selected_countries:
    filter_df = filter_df[filter_df['country'].str.contains(search_country, case=False, na=False)]

# Sorting dan display
sort_col = st.selectbox("Urutkan berdasarkan:", options=display_cols if display_cols else available_cols, index=0, key="sort_select")
sort_asc = st.checkbox("Urut menaik (A→Z)", value=False)

if sort_col in filter_df.columns:
    filter_df = filter_df.sort_values(sort_col, ascending=sort_asc, na_position='last')

cols_to_show = display_cols if display_cols else ['country','gdp_nominal','gdp_nominal_numeric','gdp_per_capita','gdp_per_capita_numeric','population_2023','gdp_growth','share_world']

# Format tampilan
display_df = filter_df[cols_to_show].head(rows).copy()

# Formatting untuk nilai numerik
if 'gdp_nominal_numeric' in display_df.columns:
    display_df['gdp_nominal_numeric'] = display_df['gdp_nominal_numeric'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
if 'gdp_per_capita_numeric' in display_df.columns:
    display_df['gdp_per_capita_numeric'] = display_df['gdp_per_capita_numeric'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
if 'gdp_growth_numeric' in display_df.columns:
    display_df['gdp_growth_numeric'] = display_df['gdp_growth_numeric'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
if 'population_2023' in display_df.columns:
    display_df['population_2023'] = display_df['population_2023'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")

st.dataframe(display_df, use_container_width=True, height=400)

st.caption(f"📊 Menampilkan {len(display_df)} dari {len(filter_df)} data yang tersaring")

# Download
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    csv = filter_df[cols_to_show].to_csv(index=False).encode('utf-8')
    st.download_button('📥 Unduh CSV', data=csv, file_name=f'gdp_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', mime='text/csv')

with col2:
    json_data = filter_df[cols_to_show].to_json(orient='records').encode('utf-8')
    st.download_button('📄 Unduh JSON', data=json_data, file_name=f'gdp_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', mime='application/json')

with col3:
    st.info("💾 Pilih format yang Anda butuhkan untuk mengunduh data", icon="ℹ️")


