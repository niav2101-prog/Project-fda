# 🌍 Global GDP Explorer Dashboard

Aplikasi Streamlit interaktif untuk analisis data ekonomi dunia dari World Bank dan UN Data.

## 📋 Fitur Utama

- 📊 Visualisasi data GDP dengan chart interaktif
- 🌐 Filter berdasarkan negara, metrik, dan top-N
- 📈 Analisis tren ekonomi global
- 📥 Download data dalam format CSV
- 🎨 Design responsif dan modern

## 🚀 Cara Deploy ke Streamlit Cloud

### 1. Push ke GitHub
```bash
git add .
git commit -m "Fix dataset path for Streamlit Cloud deployment"
git push origin main
```

### 2. Buka Streamlit Cloud
- Kunjungi https://share.streamlit.io
- Klik "New app"
- Pilih repository, branch, dan file utama: `gdp_dashboard.py`

### 3. Struktur Folder yang Benar
```
project-folder/
├── gdp_dashboard.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
    └── Global GDP Explorer 2025 (World Bank  UN Data).csv
```

## 📦 Dependencies

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.14.0
- pycountry >= 23.0.0

## 🔧 Perbaikan untuk Error "Dataset Not Found"

**Masalah:** Path relatif dengan `Path(__file__).parent` tidak bekerja di Streamlit Cloud

**Solusi:** Menggunakan `os.path.abspath()` untuk mendapatkan path absolut yang kompatibel:

```python
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, "data", "Global GDP Explorer 2025 (World Bank  UN Data).csv")
```

## ⚠️ Checklist Sebelum Deploy

- ✅ Folder `data/` sudah ada dengan file CSV
- ✅ File `requirements.txt` sudah lengkap dengan versi
- ✅ File `gdp_dashboard.py` menggunakan path absolut
- ✅ File `.gitignore` sudah ter-setup
- ✅ Semua file sudah di-commit dan di-push ke GitHub

## 🐛 Troubleshooting

**Error: "FileNotFoundError: Global GDP Explorer..."**
- Pastikan folder `data/` dan file CSV ada di repository
- Periksa apakah nama file persis sama (case-sensitive)
- Pastikan menggunakan `os.path.join()` bukan `/`

**Error: "Module not found"**
- Update `requirements.txt` dengan versi terbaru
- Tunggu 1-2 menit untuk Streamlit Cloud membangun ulang

## 📊 Data Source

- **World Bank GDP Data**: https://data.worldbank.org/
- **UN Data**: https://data.un.org/

---

**Last Updated:** January 6, 2026
