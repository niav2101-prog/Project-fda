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
def load_gdp(path: Path):
    if not path.exists():
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

# Load dataset
DATA_PATH = Path(__file__).parent / "data/Global GDP Explorer 2025 (World Bank  UN Data).csv"
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


