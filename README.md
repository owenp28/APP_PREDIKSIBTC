# Bitcoin Price Prediction App

Aplikasi prediksi harga Bitcoin berbasis machine learning yang dibangun dengan Streamlit. Aplikasi ini menyediakan analisis teknikal, prediksi harga, dan rekomendasi investasi berdasarkan data historis Bitcoin dengan pembaruan real-time.

![Bitcoin Price Prediction App](https://img.shields.io/badge/Bitcoin-Price%20Prediction-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![Real-time](https://img.shields.io/badge/Updates-Real--time-green)

## Fitur Utama

- **Analisis Harga Real-time**: Menampilkan data harga Bitcoin terkini dari API CoinMarketCap dan CoinGecko
- **Prediksi Harga**: Menggunakan machine learning untuk memprediksi harga Bitcoin di masa depan
- **Analisis Teknikal**: Menghitung indikator seperti RSI, MACD, dan Moving Averages
- **Analisis Lanjutan**: Deteksi pola harga, analisis order flow, dan profil volume
- **Backtest Otomatis**: Pengujian strategi trading dengan data real-time yang terus diperbarui
- **Rekomendasi Investasi**: Memberikan saran beli/jual berdasarkan analisis teknikal dan prediksi
- **Kalkulator Investasi**: Menghitung potensi keuntungan berdasarkan jumlah investasi
- **Target Harga**: Menyediakan target harga untuk beli, jual, dan stop loss
- **Analisis Portfolio**: Membandingkan kinerja Bitcoin dengan aset lainnya

## Struktur Proyek

```
app_prediksibtc/
├── app_prediksibtc.py     # File utama aplikasi Streamlit
├── data_load.py           # Modul untuk memuat dan memproses data Bitcoin
├── train_model.py         # Modul untuk melatih model machine learning
├── prediction_btc.py      # Modul untuk membuat prediksi harga
├── backtest.py            # Modul untuk pengujian strategi trading
├── advanced_analysis.py   # Modul untuk analisis teknikal lanjutan
├── price_patterns.py      # Modul untuk deteksi pola harga
├── order_flow.py          # Modul untuk analisis order flow
├── trading_signals.py     # Modul untuk menghasilkan sinyal trading
├── portfolio_analysis.py  # Modul untuk analisis portfolio
├── technical_analysis.py  # Modul untuk indikator teknikal
├── requirements.txt       # Daftar library yang diperlukan
├── run_app.bat            # Script batch untuk menjalankan aplikasi di Windows
├── setup.bat              # Script batch untuk setup awal
└── README.md              # Dokumentasi proyek
```

## Cara Kerja

1. Aplikasi mengambil data harga Bitcoin dari API CoinMarketCap/CoinGecko secara real-time
2. Indikator teknikal dan pola harga dihitung untuk mengidentifikasi tren pasar
3. Analisis order flow dan profil volume digunakan untuk mendeteksi tekanan beli/jual
4. Model machine learning dilatih menggunakan pola harga historis
5. Model memprediksi harga masa depan berdasarkan pola tersebut
6. Strategi trading diuji dengan backtest yang diperbarui secara real-time
7. Rekomendasi investasi dibuat berdasarkan kombinasi semua analisis

## Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama
- **Streamlit**: Framework untuk membangun antarmuka web
- **Pandas**: Manipulasi dan analisis data
- **NumPy**: Komputasi numerik
- **Scikit-learn**: Library machine learning untuk model prediksi
- **Plotly**: Visualisasi data interaktif
- **Requests**: Mengambil data dari API CoinMarketCap dan CoinGecko
- **SciPy**: Analisis statistik dan pemrosesan sinyal

## Cara Menjalankan Aplikasi

### Prasyarat

- Python 3.7 atau lebih baru
- Pip (Python package manager)

### Instalasi

1. Clone atau download repository ini
2. Buka terminal/command prompt di direktori proyek
3. Install library yang diperlukan:

```bash
pip install -r requirements.txt
```

atau

```bash
pip install streamlit pandas numpy scikit-learn plotly requests scipy
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
- Mengambil data Bitcoin dari API CoinMarketCap dan CoinGecko
- Menghitung indikator teknikal (RSI, MACD, Moving Averages)
- Menyediakan data fallback jika API tidak tersedia
- Memperbarui data secara real-time

### 2. Advanced Analysis (advanced_analysis.py)
- Deteksi pola harga seperti double top/bottom dan head & shoulders
- Analisis profil volume untuk mengidentifikasi level support/resistance
- Analisis order flow untuk mendeteksi tekanan beli/jual
- Backtest strategi trading

### 3. Backtest (backtest.py)
- Pengujian strategi trading dengan data historis
- Perhitungan metrik kinerja (total return, drawdown, win rate)
- Pembaruan real-time untuk hasil backtest
- Kombinasi sinyal dari berbagai sumber analisis

### 4. Model Training (train_model.py)
- Memproses data untuk pelatihan model
- Membuat fitur lag untuk analisis deret waktu
- Melatih model Random Forest untuk prediksi harga

### 5. Price Prediction (prediction_btc.py)
- Membuat prediksi harga untuk periode yang ditentukan
- Menggunakan model yang telah dilatih untuk memprediksi harga masa depan

### 6. Trading Signals (trading_signals.py)
- Mengkombinasikan sinyal dari berbagai analisis
- Menghasilkan rekomendasi beli/jual
- Menentukan target harga dan stop loss

### 7. Web Interface (app_prediksibtc.py)
- Antarmuka Streamlit interaktif dengan auto-refresh
- Visualisasi data dan prediksi
- Kalkulator investasi dan rekomendasi
- Tampilan backtest dan analisis portfolio

## Disclaimer

Aplikasi ini menyediakan prediksi berdasarkan data historis dan tidak boleh dianggap sebagai nasihat keuangan. Investasi cryptocurrency memiliki risiko pasar yang tinggi. Selalu lakukan riset Anda sendiri sebelum berinvestasi.

## Pengembangan Lebih Lanjut

Beberapa ide untuk pengembangan lebih lanjut:
- Integrasi dengan lebih banyak sumber data
- Implementasi model deep learning (LSTM, Transformer)
- Penambahan lebih banyak indikator teknikal dan pola harga
- Fitur notifikasi untuk perubahan harga signifikan
- Analisis sentimen dari berita dan media sosial
- Optimasi parameter strategi trading secara otomatis
- Integrasi dengan API exchange untuk trading otomatis
- Dashboard performa portfolio yang lebih komprehensif

---
