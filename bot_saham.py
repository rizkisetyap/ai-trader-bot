"""
AI HYBRID TRADER BOT — IDX Banking Sector
Dijalankan sebagai one-shot script via GitHub Actions
Jadwal: Senin–Jumat pukul 16:30 WIB (09:30 UTC)

Arsitektur:
  - Teknikal  : RandomForest (scikit-learn) + pandas-ta indicators
  - Sentimen  : Gemini 1.5 Flash (Google AI)
  - Notifikasi: Telegram Bot API
  - Scheduler : GitHub Actions cron (bukan APScheduler)
"""

import os
import sys
import logging
import time
import joblib
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import google.generativeai as genai
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pytz

# ──────────────────────────────────────────
# LOGGING  (stdout → tampil di GH Actions log)
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────
# ENV  (GitHub Secrets di-inject sebagai env var)
# ──────────────────────────────────────────
load_dotenv()  # no-op di GH Actions, berguna untuk local dev

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")

if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, GEMINI_API_KEY]):
    log.critical("❌ Kredensial API tidak lengkap! Pastikan GitHub Secrets sudah diset.")
    sys.exit(1)

# ──────────────────────────────────────────
# KONSTANTA
# ──────────────────────────────────────────
WIB          = pytz.timezone("Asia/Jakarta")
DAFTAR_SAHAM = ["BBCA.JK", "BBNI.JK", "BBRI.JK", "BMRI.JK"]
FITUR        = [
    "SMA_20", "SMA_50", "RSI_14",
    "MACD", "MACD_Signal",
    "BB_Lower", "BB_Upper",
    "Open", "Close", "Volume",
]
MODEL_DIR    = Path("model_cache")
MODEL_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════
# 1. TELEGRAM  – kirim dengan retry
# ══════════════════════════════════════════
def kirim_telegram(pesan: str, retries: int = 3) -> bool:
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": pesan, "parse_mode": "Markdown"}
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
            log.info("✅ Telegram: pesan terkirim.")
            return True
        except requests.RequestException as e:
            log.warning(f"⚠️  Telegram attempt {attempt}/{retries} gagal: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)   # back-off: 2s, 4s
    log.error("❌ Telegram: gagal kirim setelah semua retry.")
    return False


# ══════════════════════════════════════════
# 2. SENTIMEN  – Gemini AI dengan in-run cache
# ══════════════════════════════════════════
class SentimenAnalyser:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self._llm   = genai.GenerativeModel("gemini-1.5-flash")
        self._cache: dict[str, tuple[str, str]] = {}

    def analisa(self, ticker_obj, kode: str) -> tuple[str, str]:
        if kode in self._cache:
            return self._cache[kode]

        try:
            berita = ticker_obj.news or []
            if not berita:
                return self._simpan(kode, "NETRAL ⚪", "Tidak ada berita hari ini.")

            judul = "\n".join(f"- {b['title']}" for b in berita[:5])
            prompt = f"""Kamu adalah analis saham senior Indonesia (pasar IDX).
Baca 5 judul berita terbaru saham {kode}:
{judul}

Apakah sentimen berita secara keseluruhan Positif, Negatif, atau Netral terhadap harga saham?
Jawab HANYA dengan format (tanpa teks lain):
[STATUS] | [Alasan singkat 1 kalimat Bahasa Indonesia]
Contoh: POSITIF | Laba bersih Q2 naik 15% melampaui ekspektasi analis."""

            respon = self._llm.generate_content(prompt)
            teks   = respon.text.strip()

            if "|" not in teks:
                return self._simpan(kode, "NETRAL ⚪", "Format jawaban AI tidak sesuai.")

            status_raw, alasan = teks.split("|", 1)
            status_raw = status_raw.strip().upper()

            if "POSITIF" in status_raw:
                status = "POSITIF 🟢"
            elif "NEGATIF" in status_raw:
                status = "NEGATIF 🔴"
            else:
                status = "NETRAL ⚪"

            return self._simpan(kode, status, alasan.strip())

        except Exception as e:
            log.warning(f"Sentimen {kode} error: {e}")
            return self._simpan(kode, "NETRAL ⚪", "Gagal menganalisis sentimen berita.")

    def _simpan(self, kode, status, alasan):
        self._cache[kode] = (status, alasan)
        return status, alasan


# ══════════════════════════════════════════
# 3. MODEL MANAGER  – selalu retrain di GH Actions
#    (runner ephemeral, tidak ada cache antar run)
# ══════════════════════════════════════════
class ModelManager:
    def __init__(self, kode: str):
        self.kode   = kode
        self._model = None

    def train(self, X_train, y_train, X_test, y_test) -> "ModelManager":
        log.info(f"  [{self.kode}] Melatih RandomForest...")
        self._model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X_train, y_train)
        acc = accuracy_score(y_test, self._model.predict(X_test))
        log.info(f"  [{self.kode}] Akurasi validasi: {acc:.2%}")
        return self

    def prediksi(self, X: pd.DataFrame) -> tuple[int, float]:
        label     = self._model.predict(X)[0]
        proba     = self._model.predict_proba(X)[0]
        keyakinan = proba[1] * 100 if label == 1 else proba[0] * 100
        return int(label), round(keyakinan, 1)


# ══════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════
def buat_fitur(df: pd.DataFrame) -> pd.DataFrame:
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
    df["Target"]      = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    return df


# ══════════════════════════════════════════
# 5. PIPELINE PER SAHAM
# ══════════════════════════════════════════
def analisa_satu_saham(kode: str, sentimen: SentimenAnalyser) -> str | None:
    log.info(f"  [{kode}] Mengambil data historis (5 tahun)...")
    try:
        saham = yf.Ticker(kode)
        df    = saham.history(period="5y")
    except Exception as e:
        log.error(f"  [{kode}] Gagal ambil data yfinance: {e}")
        return None

    if df.empty or len(df) < 100:
        log.warning(f"  [{kode}] Data tidak cukup, skip.")
        return None

    df = buat_fitur(df)

    # ⚠️  Simpan baris terakhir (hari ini) SEBELUM dropna
    #     agar tidak kehilangan baris paling baru untuk prediksi
    data_hari_ini = df.iloc[-1:].copy()

    df_bersih = df.dropna()
    if len(df_bersih) < 60:
        log.warning(f"  [{kode}] Data bersih kurang dari 60 baris, skip.")
        return None

    X = df_bersih[FITUR]
    y = df_bersih["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    try:
        mgr = ModelManager(kode).train(X_train, y_train, X_test, y_test)
    except Exception as e:
        log.error(f"  [{kode}] Gagal train model: {e}")
        return None

    X_pred = data_hari_ini[FITUR]
    if X_pred.isnull().values.any():
        log.warning(f"  [{kode}] Data hari ini mengandung NaN, skip prediksi.")
        return None

    label, keyakinan    = mgr.prediksi(X_pred)
    status_teknikal     = "NAIK 🟢" if label == 1 else "TURUN/DATAR 🔴"
    harga_tutup         = data_hari_ini["Close"].values[0]

    log.info(f"  [{kode}] Menganalisis sentimen berita...")
    status_sentimen, alasan_sentimen = sentimen.analisa(saham, kode)

    baris = (
        f"🏢 *{kode}* (Rp {harga_tutup:,.0f})\n"
        f"📊 *Teknikal:* {status_teknikal} ({keyakinan:.1f}%)\n"
        f"📰 *Sentimen:* {status_sentimen}\n"
        f"💬 _{alasan_sentimen}_\n"
        f"─────────────────────\n"
    )
    log.info(f"  [{kode}] Selesai → {status_teknikal} | {status_sentimen}")
    return baris


# ══════════════════════════════════════════
# 6. MAIN — one-shot, dipanggil oleh GH Actions
# ══════════════════════════════════════════
def main() -> int:
    waktu_mulai = datetime.now(WIB)
    log.info("=" * 55)
    log.info(f"AI HYBRID TRADER BOT — {waktu_mulai.strftime('%A, %d %B %Y %H:%M WIB')}")
    log.info("=" * 55)

    sentimen = SentimenAnalyser()

    header = (
        "🤖 *LAPORAN AI TRADER BOT (HYBRID)* 🤖\n"
        "_Teknikal (RandomForest) + Sentimen (Gemini AI)_\n"
        f"📅 {waktu_mulai.strftime('%d %B %Y, %H:%M WIB')}\n\n"
    )

    baris_laporan: list[str] = []
    gagal:         list[str] = []

    for kode in DAFTAR_SAHAM:
        log.info(f"▶ ANALISA: {kode}")
        baris = analisa_satu_saham(kode, sentimen)
        if baris:
            baris_laporan.append(baris)
        else:
            gagal.append(kode)

    if not baris_laporan:
        log.error("Tidak ada saham yang berhasil dianalisis. Batalkan pengiriman.")
        return 1

    laporan  = header + "\n".join(baris_laporan)
    if gagal:
        laporan += f"\n⚠️ Gagal diproses: {', '.join(gagal)}"
    laporan += "\n_Bot by AI Hybrid Trader — IDX Banking_"

    kirim_telegram(laporan)

    durasi = round((datetime.now(WIB) - waktu_mulai).total_seconds())
    log.info(f"✅ SELESAI dalam {durasi} detik.")
    log.info("=" * 55)
    return 0


if __name__ == "__main__":
    sys.exit(main())
