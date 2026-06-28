"""
AI HYBRID TRADER BOT — IDX Banking Sector
Optimized for production use with scheduled execution (Mon–Fri 16:00 WIB)

Improvements:
- Cron scheduler (Mon–Fri 16:00 WIB) via APScheduler
- Modular, class-based architecture
- Robust error handling & retry logic for API calls
- Model retraining logic (weekly auto-retrain)
- Feature engineering improvements
- Sentiment caching (avoid repeated API calls per run)
- Unified logging to file + console
- Graceful shutdown handling
"""

import os
import sys
import logging
import time
import signal
import joblib
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import google.generativeai as genai
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# ==========================================
# SETUP LOGGING
# ==========================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "trader_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ==========================================
# LOAD ENV
# ==========================================
load_dotenv()

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")

if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, GEMINI_API_KEY]):
    log.critical("Kredensial API tidak lengkap! Cek file .env Anda.")
    sys.exit(1)

# ==========================================
# CONSTANTS
# ==========================================
DAFTAR_SAHAM = ["BBCA.JK", "BBNI.JK", "BBRI.JK", "BMRI.JK"]
FITUR        = [
    "SMA_20", "SMA_50", "RSI_14",
    "MACD", "MACD_Signal",
    "BB_Lower", "BB_Upper",
    "Open", "Close", "Volume",
]
MODEL_DIR    = Path("model_cache")
MODEL_DIR.mkdir(exist_ok=True)

# Retrain jika model lebih tua dari N hari
MODEL_MAX_AGE_DAYS = 7

# Timezone WIB (Jakarta)
WIB = pytz.timezone("Asia/Jakarta")


# ==========================================
# TELEGRAM HELPER
# ==========================================
def kirim_telegram(pesan: str, retries: int = 3) -> bool:
    """Kirim pesan Telegram dengan retry logic."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": pesan,
        "parse_mode": "Markdown",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            log.info("Telegram: pesan terkirim.")
            return True
        except requests.RequestException as e:
            log.warning(f"Telegram attempt {attempt}/{retries} gagal: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)   # exponential back-off
    log.error("Telegram: gagal kirim setelah semua retry.")
    return False


# ==========================================
# SENTIMENT ANALYSER
# ==========================================
class SentimenAnalyser:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self._model = genai.GenerativeModel("gemini-1.5-flash")
        self._cache: dict[str, tuple[str, str]] = {}   # ticker -> (status, alasan)

    def analisa(self, ticker_obj, kode: str) -> tuple[str, str]:
        """Return (status_emoji, alasan). Cached per run."""
        if kode in self._cache:
            return self._cache[kode]

        try:
            berita = ticker_obj.news
            if not berita:
                result = ("NETRAL ⚪", "Tidak ada berita hari ini.")
                self._cache[kode] = result
                return result

            judul_berita = "\n".join(
                [f"- {b['title']}" for b in berita[:5]]   # 5 judul untuk akurasi lebih baik
            )
            prompt = f"""
Kamu adalah analis saham senior Indonesia yang berpengalaman di pasar IDX.
Baca 5 judul berita terbaru mengenai saham {kode} berikut:
{judul_berita}

Apakah sentimen berita secara keseluruhan Positif, Negatif, atau Netral terhadap harga saham?
Jawab hanya dengan format: [STATUS] | [Alasan singkat 1 kalimat, dalam Bahasa Indonesia]
Contoh: POSITIF | Laba bersih Q2 naik 15% melebihi ekspektasi analis.
"""
            respon = self._model.generate_content(prompt)
            teks = respon.text.strip()

            if "|" in teks:
                status_raw, alasan = teks.split("|", 1)
                status_raw = status_raw.strip().upper()
                alasan = alasan.strip()

                if "POSITIF" in status_raw:
                    status = "POSITIF 🟢"
                elif "NEGATIF" in status_raw:
                    status = "NEGATIF 🔴"
                else:
                    status = "NETRAL ⚪"
            else:
                status, alasan = "NETRAL ⚪", "Format jawaban AI tidak sesuai."

        except Exception as e:
            log.warning(f"Sentimen {kode} error: {e}")
            status, alasan = "NETRAL ⚪", "Gagal menganalisis sentimen berita."

        result = (status, alasan)
        self._cache[kode] = result
        return result


# ==========================================
# MODEL MANAGER
# ==========================================
class ModelManager:
    def __init__(self, kode: str):
        self.kode      = kode
        self.path      = MODEL_DIR / f"model_{kode}.joblib"
        self._model    = None

    def _perlu_retrain(self) -> bool:
        if not self.path.exists():
            return True
        umur_hari = (time.time() - self.path.stat().st_mtime) / 86400
        return umur_hari >= MODEL_MAX_AGE_DAYS

    def load_atau_train(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> "ModelManager":
        if not self._perlu_retrain():
            log.info(f"  [{self.kode}] Memuat model dari cache.")
            self._model = joblib.load(self.path)
        else:
            log.info(f"  [{self.kode}] Melatih model baru (RandomForest)...")
            self._model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,       # pakai semua CPU core
            )
            self._model.fit(X_train, y_train)
            acc = accuracy_score(y_test, self._model.predict(X_test))
            log.info(f"  [{self.kode}] Akurasi validasi: {acc:.2%}")
            joblib.dump(self._model, self.path)
            log.info(f"  [{self.kode}] Model disimpan → {self.path}")
        return self

    def prediksi(self, X: pd.DataFrame) -> tuple[int, float]:
        """Return (label, keyakinan_persen)."""
        label = self._model.predict(X)[0]
        proba = self._model.predict_proba(X)[0]
        keyakinan = proba[1] * 100 if label == 1 else proba[0] * 100
        return int(label), round(keyakinan, 1)


# ==========================================
# FEATURE ENGINEERING
# ==========================================
def buat_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan semua indikator teknikal ke dataframe."""
    df = df.copy()

    df["SMA_20"]      = ta.sma(df["Close"], length=20)
    df["SMA_50"]      = ta.sma(df["Close"], length=50)
    df["RSI_14"]      = ta.rsi(df["Close"], length=14)

    macd              = ta.macd(df["Close"])
    df["MACD"]        = macd.iloc[:, 0]
    df["MACD_Signal"] = macd.iloc[:, 1]

    bbands            = ta.bbands(df["Close"])
    df["BB_Lower"]    = bbands.iloc[:, 0]
    df["BB_Upper"]    = bbands.iloc[:, 2]

    # Target: 1 = besok naik, 0 = turun/flat
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    return df


# ==========================================
# CORE ANALYSIS PER SAHAM
# ==========================================
def analisa_satu_saham(kode: str, sentimen_analyser: SentimenAnalyser) -> str | None:
    """
    Jalankan full pipeline untuk 1 saham.
    Return: string baris laporan, atau None jika gagal.
    """
    log.info(f"  [{kode}] Mengambil data historis...")
    try:
        saham = yf.Ticker(kode)
        df    = saham.history(period="5y")
    except Exception as e:
        log.error(f"  [{kode}] Gagal ambil data yfinance: {e}")
        return None

    if df.empty or len(df) < 100:
        log.warning(f"  [{kode}] Data tidak cukup, skip.")
        return None

    # Feature engineering
    df = buat_fitur(df)

    # Simpan data hari ini SEBELUM dropna (agar prediksi tidak kehilangan baris terakhir)
    data_hari_ini = df.iloc[-1:].copy()

    df_bersih = df.dropna()
    if len(df_bersih) < 60:
        log.warning(f"  [{kode}] Data bersih tidak cukup setelah dropna, skip.")
        return None

    X = df_bersih[FITUR]
    y = df_bersih["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Load / train model
    try:
        mgr = ModelManager(kode).load_atau_train(X_train, y_train, X_test, y_test)
    except Exception as e:
        log.error(f"  [{kode}] Gagal train/load model: {e}")
        return None

    # Prediksi
    X_pred = data_hari_ini[FITUR]
    if X_pred.isnull().values.any():
        log.warning(f"  [{kode}] Data hari ini mengandung NaN, skip prediksi.")
        return None

    label, keyakinan = mgr.prediksi(X_pred)
    status_teknikal  = "NAIK 🟢" if label == 1 else "TURUN/DATAR 🔴"
    harga_tutup      = data_hari_ini["Close"].values[0]

    # Analisa sentimen
    log.info(f"  [{kode}] Menganalisis sentimen berita...")
    status_sentimen, alasan_sentimen = sentimen_analyser.analisa(saham, kode)

    # Susun baris laporan
    baris = (
        f"🏢 *{kode}* (Rp {harga_tutup:,.0f})\n"
        f"📊 *Teknikal:* {status_teknikal} ({keyakinan:.1f}%)\n"
        f"📰 *Sentimen:* {status_sentimen}\n"
        f"💬 _{alasan_sentimen}_\n"
        f"─────────────────────\n"
    )
    log.info(f"  [{kode}] Selesai → {status_teknikal} | {status_sentimen}")
    return baris


# ==========================================
# MAIN JOB — dijalankan tiap hari
# ==========================================
def jalankan_analisis():
    waktu_mulai = datetime.now(WIB)
    log.info(f"{'='*50}")
    log.info(f"MEMULAI ANALISIS — {waktu_mulai.strftime('%A, %d %B %Y %H:%M WIB')}")
    log.info(f"{'='*50}")

    sentimen_analyser = SentimenAnalyser()

    header = (
        "🤖 *LAPORAN AI TRADER BOT (HYBRID)* 🤖\n"
        "_Teknikal (RandomForest) + Sentimen (Gemini AI)_\n"
        f"📅 {waktu_mulai.strftime('%d %B %Y, %H:%M WIB')}\n\n"
    )

    baris_laporan: list[str] = []
    gagal: list[str] = []

    for kode in DAFTAR_SAHAM:
        log.info(f"▶ ANALISA: {kode}")
        baris = analisa_satu_saham(kode, sentimen_analyser)
        if baris:
            baris_laporan.append(baris)
        else:
            gagal.append(kode)

    laporan = header + "\n".join(baris_laporan)

    if gagal:
        laporan += f"\n⚠️ Gagal diproses: {', '.join(gagal)}"

    laporan += "\n_Bot by AI Hybrid Trader — IDX Banking_"

    log.info("Mengirim laporan ke Telegram...")
    kirim_telegram(laporan)

    durasi = (datetime.now(WIB) - waktu_mulai).seconds
    log.info(f"ANALISIS SELESAI dalam {durasi} detik.")
    log.info(f"{'='*50}\n")


# ==========================================
# SCHEDULER — Senin–Jumat 16:00 WIB
# ==========================================
def mulai_scheduler():
    scheduler = BlockingScheduler(timezone=WIB)

    scheduler.add_job(
        jalankan_analisis,
        trigger=CronTrigger(
            day_of_week="mon-fri",   # Senin–Jumat
            hour=16,
            minute=0,
            timezone=WIB,
        ),
        id="analisis_harian",
        name="AI Hybrid Trader — Analisis Harian 16:00 WIB",
        misfire_grace_time=300,      # toleransi misfired 5 menit
        coalesce=True,               # jangan jalankan ulang job yang menumpuk
    )

    log.info("Scheduler aktif — Analisis dijadwalkan Senin–Jumat pukul 16:00 WIB.")
    log.info("Tekan Ctrl+C untuk menghentikan.\n")

    # Graceful shutdown
    def _shutdown(sig, frame):
        log.info("Menerima sinyal shutdown, menghentikan scheduler...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    scheduler.start()


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Hybrid Trader Bot — IDX Banking")
    parser.add_argument(
        "--now",
        action="store_true",
        help="Jalankan analisis sekarang (tanpa menunggu jadwal)",
    )
    args = parser.parse_args()

    if args.now:
        log.info("Mode --now: menjalankan analisis segera...")
        jalankan_analisis()
    else:
        mulai_scheduler()
