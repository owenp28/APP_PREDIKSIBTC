# Bitcoin Price Prediction App

Aplikasi prediksi harga Bitcoin berbasis machine learning yang dibangun dengan Streamlit. Aplikasi ini menyediakan analisis teknikal, prediksi harga, dan rekomendasi investasi berdasarkan data historis Bitcoin.

![Bitcoin Price Prediction App](https://img.shields.io/badge/Bitcoin-Price%20Prediction-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.7+-blue)

## Fitur Utama

- **Analisis Harga Real-time**: Menampilkan data harga Bitcoin terkini dari API CoinGecko
- **Prediksi Harga**: Menggunakan machine learning untuk memprediksi harga Bitcoin di masa depan
- **Analisis Teknikal**: Menghitung indikator seperti RSI, MACD, dan Moving Averages
- **Rekomendasi Investasi**: Memberikan saran beli/jual berdasarkan analisis teknikal dan prediksi
- **Kalkulator Investasi**: Menghitung potensi keuntungan berdasarkan jumlah investasi
- **Target Harga**: Menyediakan target harga untuk beli, jual, dan stop loss

## Struktur Proyek

```
app_prediksibtc/
├── app_prediksibtc.py     # File utama aplikasi Streamlit
├── data_load.py           # Modul untuk memuat dan memproses data Bitcoin
├── train_model.py         # Modul untuk melatih model machine learning
├── prediction_btc.py      # Modul untuk membuat prediksi harga
├── run_app.bat            # Script batch untuk menjalankan aplikasi di Windows
└── README.md              # Dokumentasi proyek
```

## Cara Kerja

1. Aplikasi mengambil data harga Bitcoin dari API CoinGecko
2. Indikator teknikal dihitung untuk mengidentifikasi tren pasar
3. Model machine learning dilatih menggunakan pola harga historis
4. Model memprediksi harga masa depan berdasarkan pola tersebut
5. Rekomendasi investasi dibuat berdasarkan analisis teknikal dan prediksi harga

## Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama
- **Streamlit**: Framework untuk membangun antarmuka web
- **Pandas**: Manipulasi dan analisis data
- **Scikit-learn**: Library machine learning untuk model prediksi
- **Matplotlib & Plotly**: Visualisasi data interaktif
- **Requests**: Mengambil data dari API CoinGecko

## Cara Menjalankan Aplikasi

### Prasyarat

- Python 3.7 atau lebih baru
- Pip (Python package manager)

### Instalasi

1. Clone atau download repository ini
2. Buka terminal/command prompt di direktori proyek
3. Install library yang diperlukan:

```bash
pip install streamlit pandas matplotlib numpy scikit-learn plotly requests
```

### Menjalankan Aplikasi

#### Di Windows:
- Klik dua kali pada file `run_app.bat`

atau

- Buka terminal/command prompt di direktori proyek
- Jalankan perintah:

```bash
streamlit run app_prediksibtc.py
```

#### Di macOS/Linux:
- Buka terminal di direktori proyek
- Jalankan perintah:

```bash
streamlit run app_prediksibtc.py
```

Aplikasi akan terbuka di browser web Anda secara otomatis.

## Komponen Utama

### 1. Data Loading (data_load.py)
- Mengambil data Bitcoin dari API CoinGecko
- Menghitung indikator teknikal (RSI, MACD, Moving Averages)
- Menyediakan data fallback jika API tidak tersedia

### 2. Model Training (train_model.py)
- Memproses data untuk pelatihan model
- Membuat fitur lag untuk analisis deret waktu
- Melatih model Random Forest untuk prediksi harga

### 3. Price Prediction (prediction_btc.py)
- Membuat prediksi harga untuk periode yang ditentukan
- Menggunakan model yang telah dilatih untuk memprediksi harga masa depan

### 4. Web Interface (app_prediksibtc.py)
- Antarmuka Streamlit interaktif
- Visualisasi data dan prediksi
- Kalkulator investasi dan rekomendasi

## Disclaimer

Aplikasi ini menyediakan prediksi berdasarkan data historis dan tidak boleh dianggap sebagai nasihat keuangan. Investasi cryptocurrency memiliki risiko pasar yang tinggi. Selalu lakukan riset Anda sendiri sebelum berinvestasi.

## Pengembangan Lebih Lanjut

Beberapa ide untuk pengembangan lebih lanjut:
- Integrasi dengan lebih banyak sumber data
- Implementasi model machine learning yang lebih canggih
- Penambahan lebih banyak indikator teknikal
- Fitur notifikasi untuk perubahan harga signifikan
- Analisis sentimen dari berita dan media sosial

---
