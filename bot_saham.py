from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import joblib
import requests
import google.generativeai as genai  # MODUL BARU: AI Pembaca Teks (Otak Kanan)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Memuat variabel dari file .env
load_dotenv()

# ==========================================
# KONFIGURASI API (DARI ENV)
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validasi keamanan sederhana
if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, GEMINI_API_KEY]):
    raise ValueError("Kredensial API tidak lengkap! Cek file .env Anda.")
# Setup Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model_ai_teks = genai.GenerativeModel('gemini-1.5-flash') # Model yang cepat untuk baca teks

def kirim_notifikasi_telegram(pesan):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": pesan, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"-> Gagal kirim Telegram: {e}")

# ==========================================
# FUNGSI OTAK KANAN: ANALISA SENTIMEN
# ==========================================
def analisa_sentimen_berita(saham_ticker, nama_saham):
    try:
        # 1. Tarik berita terbaru dari yfinance
        berita = saham_ticker.news
        if not berita:
            return "NETRAL ⚪", "Tidak ada berita hari ini."
        
        # 2. Gabungkan 3 judul berita terbaru
        kumpulan_judul = "\n".join([f"- {b['title']}" for b in berita[:3]])
        
        # 3. Prompt (Perintah) untuk AI
        prompt = f"""
        Kamu adalah analis saham profesional. Baca 3 judul berita terbaru mengenai saham {nama_saham} berikut:
        {kumpulan_judul}
        
        Apakah sentimen berita tersebut secara keseluruhan Positif, Negatif, atau Netral untuk pergerakan harga saham? 
        Jawab hanya dengan format: [STATUS] | [Alasan sangat singkat 1 kalimat]
        Contoh: POSITIF | Laba perusahaan meningkat tajam kuartal ini.
        """
        
        # 4. Minta AI berpikir dan menjawab
        respon = model_ai_teks.generate_content(prompt)
        teks_jawaban = respon.text.strip()
        
        # Ekstrak Status dan Alasan
        if "|" in teks_jawaban:
            status, alasan = teks_jawaban.split("|", 1)
            status = status.strip().upper()
            alasan = alasan.strip()
            
            # Tambahkan Emoji
            if "POSITIF" in status: status = "POSITIF 🟢"
            elif "NEGATIF" in status: status = "NEGATIF 🔴"
            else: status = "NETRAL ⚪"
            
            return status, alasan
        else:
            return "NETRAL ⚪", "AI gagal menyimpulkan format."
            
    except Exception as e:
        return "NETRAL ⚪", f"Error membaca sentimen."

# ==========================================
# PROGRAM UTAMA
# ==========================================
print("=== MEMULAI SISTEM AI HYBRID (TEKNIKAL & SENTIMEN) ===")

daftar_saham = ["BBCA.JK", "BBNI.JK", "BBRI.JK", "BMRI.JK"]
nama_file_jurnal = "jurnal_trading_ai_final.csv"

teks_laporan = "🤖 *LAPORAN AI TRADER BOT (HYBRID)* 🤖\n"
teks_laporan += "Menggabungkan Teknikal & Sentimen Berita\n\n"

for kode in daftar_saham:
    print(f"\nMENGANALISA: {kode}...")
    
    # --- PROSES OTAK KIRI (TEKNIKAL MATEMATIKA) ---
    saham = yf.Ticker(kode)
    df = saham.history(period="5y")
    
    if df.empty:
        continue

    # Indikator
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd.iloc[:, 0]
    df['MACD_Signal'] = macd.iloc[:, 1]
    bbands = ta.bbands(df['Close'])
    df['BB_Lower'] = bbands.iloc[:, 0]
    df['BB_Upper'] = bbands.iloc[:, 2]

    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df = df.dropna()

    fitur = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Lower', 'BB_Upper', 'Open', 'Close', 'Volume']
    X = df[fitur]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    nama_file_memori = f"memori_ai_{kode}.joblib"
    if os.path.exists(nama_file_memori):
        model = joblib.load(nama_file_memori)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, nama_file_memori)

    # Prediksi Angka
    data_hari_ini = df.iloc[-1:]
    prediksi_besok = model.predict(data_hari_ini[fitur])
    probabilitas = model.predict_proba(data_hari_ini[fitur])[0]
    
    status_teknikal = "NAIK 🟢" if prediksi_besok[0] == 1 else "TURUN/DATAR 🔴"
    keyakinan = probabilitas[1] * 100 if prediksi_besok[0] == 1 else probabilitas[0] * 100
    harga_penutupan = data_hari_ini['Close'].values[0]

    # --- PROSES OTAK KANAN (SENTIMEN TEKS) ---
    print(f"-> Membaca berita {kode}...")
    status_sentimen, alasan_sentimen = analisa_sentimen_berita(saham, kode)

    # --- MENYUSUN LAPORAN TELEGRAM ---
    teks_laporan += f"🏢 *{kode}* (Rp {harga_penutupan:,.0f})\n"
    teks_laporan += f"📊 *Teknikal:* {status_teknikal} ({keyakinan:.1f}%)\n"
    teks_laporan += f"📰 *Sentimen:* {status_sentimen}\n"
    teks_laporan += f"💬 _{alasan_sentimen}_\n"
    teks_laporan += "-------------------------\n"
    print(f"-> Selesai diproses.")

print("\n=== SEMUA ANALISA SELESAI ===")
print("Mengirim Laporan ke Telegram...")
kirim_notifikasi_telegram(teks_laporan)
print("Selesai!")