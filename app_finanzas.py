# app_finanzas.py
# -------------------------------------------------------
# APP Finanzas - Bernardo Sánchez
# Tema blanco mejorado + LOGOS + estilos custom
# + Comparativa con índices
# + Proyección de precio (1 ticker)
# + Traducción automática de la descripción al español
# + Tabla mensual (últimos 5 años) de Riesgo vs Rendimiento con descarga CSV
# + ENCABEZADO: junto al logo y nombre, se muestra precio actual y cambio de hoy
# + Pestaña: Simulador de Portafolio (monto, riesgo y sector)
# + Pestaña: Análisis con IA (explicación + chat)
# + Pestaña: Tutor de Finanzas (IA) con explicador + quiz
# + Pestaña: Sentimiento & Noticias (IA)
# + Pestaña: Scanner de oportunidades (IA)
# + Pestaña: Acciones por país (top lista por mercado, con logos)
# -------------------------------------------------------

import json
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import plotly.graph_objs as go
import plotly.express as px
import requests
from io import BytesIO
from PIL import Image
from urllib.parse import urlparse
from deep_translator import GoogleTranslator  # para traducir descripciones al español
from openai import OpenAI  # IA

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="APP Finanzas - Bernardo Sánchez",
    page_icon="💼",
    layout="wide"
)


def set_custom_style():
    st.markdown("""
    <style>
    /* Fondo general de la app */
    .stApp {
        background: radial-gradient(circle at top left, #f9fafb 0, #e5e7eb 35%, #d1d5db 100%);
    }

    /* Contenedor principal como tarjeta */
    main .block-container {
        padding: 2.5rem 3rem 3rem 3rem;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        border-radius: 1.5rem;
        background-color: rgba(255, 255, 255, 0.95);
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
    }

    /* Sidebar más oscuro */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0b1120 40%, #020617 100%) !important;
        color: #e5e7eb !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #e5e7eb !important;
    }

    /* Título principal */
    h1 {
        font-weight: 800 !important;
        letter-spacing: 0.03em;
    }

    /* Subtítulos */
    h2, h3 {
        font-weight: 700 !important;
        color: #111827 !important;
    }

    /* Botones generales */
    .stButton>button {
        border-radius: 999px;
        border: none;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
        color: #f9fafb;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.45);
        transition: all 0.15s ease-in-out;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 40px rgba(37, 99, 235, 0.55);
        opacity: 0.98;
    }

    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.35);
    }

    /* Tabs tipo "pastilla" */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 0.5rem 1.4rem !important;
        background-color: transparent;
        color: #6b7280;
        font-weight: 600;
        border: 1px solid transparent;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #111827;
        color: #f9fafb;
        border-color: #111827;
    }

    /* Métricas como tarjetas */
    [data-testid="stMetric"] {
        background-color: #f9fafb;
        padding: 0.9rem 1rem;
        border-radius: 1rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    }

    [data-testid="stMetric"] label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
    }

    [data-testid="stMetric"] div {
        color: #111827;
    }

    /* Expanders más limpios */
    details {
        background-color: #f9fafb;
        border-radius: 1rem;
        padding: 0.6rem 0.9rem;
        border: 1px solid #e5e7eb;
    }

    details summary {
        font-weight: 600;
        color: #111827;
    }

    /* Dataframes con borde suave */
    .stDataFrame {
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }

    /* Inputs y selects más redondeados */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stDateInput>div>div>input,
    .stSelectbox>div>div>div {
        border-radius: 999px !important;
    }
    </style>
    """, unsafe_allow_html=True)


set_custom_style()

# -------------------- HELPERS DATA --------------------
@st.cache_data(show_spinner=False)
def fetch_history(tickers, start, end, interval="1d"):
    """
    Descarga histórico OHLCV de Yahoo Finance para los tickers dados (rango seleccionado).
    Devuelve:
      - dict_hist: dict[ticker]->DataFrame con ['Open','High','Low','Close','Adj Close','Volume']
      - closes_df: DataFrame columnas=tickers con Adj Close
    """
    dict_hist = {}
    closes = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            df = tk.history(
                start=start,
                end=end + timedelta(days=1),   # yfinance usa end EXCLUSIVO
                interval=interval,
                auto_adjust=False
            )
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                if col not in df.columns:
                    df[col] = np.nan
            df.index.name = "Date"
            df = df.dropna(how="all")
            if not df.empty:
                df["Ticker"] = t.upper()
                dict_hist[t.upper()] = df.copy()
                closes.append(df["Adj Close"].rename(t.upper()))
        except Exception:
            pass
    closes_df = pd.concat(closes, axis=1).sort_index() if closes else pd.DataFrame()
    return dict_hist, closes_df


@st.cache_data(show_spinner=False)
def fetch_full_history(ticker: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
    """Descarga histórico entre start y end para la tabla mensual de 5 años."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(
            start=start,
            end=end + timedelta(days=1),
            interval=interval,
            auto_adjust=False
        )
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if col not in df.columns:
                df[col] = np.nan
        df.index.name = "Date"
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_static_info(ticker: str):
    """Info extendida (sin precio/cambio): nombre, sector, país, mcap, website, logo_url, resumen."""
    t = yf.Ticker(ticker)
    long_name = sector = country = website = None
    market_cap = None
    summary = None
    logo_url = None
    try:
        info = t.get_info()
        long_name = info.get("longName") or info.get("shortName")
        sector = info.get("sector")
        country = info.get("country")
        market_cap = info.get("marketCap")
        summary = info.get("longBusinessSummary")
        website = info.get("website")
        logo_url = info.get("logo_url") or info.get("logoUrl")
    except Exception:
        pass
    return {
        "long_name": long_name,
        "sector": sector,
        "country": country,
        "market_cap": market_cap,
        "summary": summary,
        "website": website,
        "logo_url": logo_url
    }


@st.cache_data(show_spinner=False)
def fetch_logo_image(info_dict: dict, size: int = 48):
    """
    Intenta bajar el logo:
      1) info['logo_url'] (Yahoo)
      2) Clearbit a partir del website (https://logo.clearbit.com/<domain>)
    Devuelve PIL.Image o None.
    """
    url = info_dict.get("logo_url")
    if not url and info_dict.get("website"):
        try:
            domain = urlparse(info_dict["website"]).netloc or info_dict["website"]
            url = f"https://logo.clearbit.com/{domain}"
        except Exception:
            url = None
    if not url:
        return None
    try:
        r = requests.get(url, timeout=7)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert("RGBA")
            img.thumbnail((size, size))
            return img
    except Exception:
        return None
    return None


def latest_price_and_change(df: pd.DataFrame):
    """
    price: último Close (fallback Adj Close)
    change_pct: variación vs punto inmediato anterior (según intervalo)
    last_dt: timestamp del último dato
    """
    if df is None or df.empty:
        return None, None, None
    df2 = df.dropna(subset=["Close", "Adj Close"], how="all").copy()
    if len(df2) == 0:
        return None, None, None
    last = df2.iloc[-1]
    price = last["Close"] if not np.isnan(last.get("Close", np.nan)) else last.get("Adj Close", np.nan)
    change_pct = None
    if len(df2) >= 2:
        prev = df2.iloc[-2]
        prev_price = prev["Close"] if not np.isnan(prev.get("Close", np.nan)) else prev.get("Adj Close", np.nan)
        if prev_price and not np.isnan(prev_price) and price and not np.isnan(price):
            change_pct = (price / prev_price - 1.0) * 100.0
    last_dt = df2.index[-1]
    return price, change_pct, last_dt


@st.cache_data(show_spinner=False, ttl=300)
def fetch_today_quote(ticker: str):
    """
    Devuelve (precio_actual, cambio_pct_hoy) usando fast_info de yfinance.
    Si no está disponible, calcula con history(period='2d', interval='1d').
    """
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", None)
        if fi:
            last = fi.get("last_price") or fi.get("lastPrice")
            prev = fi.get("previous_close") or fi.get("previousClose")
            if last is not None and prev is not None and prev != 0:
                return float(last), (float(last) / float(prev) - 1.0) * 100.0
    except Exception:
        pass
    try:
        dfq = yf.Ticker(ticker).history(period="2d", interval="1d", auto_adjust=False).dropna(subset=["Close"])
        if len(dfq) >= 2:
            last = float(dfq["Close"].iloc[-1])
            prev = float(dfq["Close"].iloc[-2])
            chg = (last / prev - 1.0) * 100.0 if prev != 0 else None
            return last, chg
        elif len(dfq) == 1:
            return float(dfq["Close"].iloc[-1]), None
    except Exception:
        pass
    return None, None


def annualized_metrics(series: pd.Series, rf_annual: float = 0.0):
    """Retorno anualizado, volatilidad anualizada y Sharpe (a partir de precios)."""
    s = series.dropna()
    if len(s) < 2:
        return np.nan, np.nan, np.nan
    rets = s.pct_change().dropna()
    n = len(rets)
    total_return = s.iloc[-1] / s.iloc[0] - 1
    ann_return = (1 + total_return) ** (252 / max(n, 1)) - 1
    ann_vol = rets.std(ddof=0) * np.sqrt(252)
    sharpe = (ann_return - rf_annual) / ann_vol if ann_vol and ann_vol > 0 else np.nan
    return ann_return, ann_vol, sharpe


def rolling_vol(series: pd.Series, window: int = 21):
    r = series.dropna().pct_change().dropna()
    return r.rolling(window).std() * np.sqrt(252)


def rolling_sharpe(series: pd.Series, rf_annual: float, window: int = 63):
    r = series.dropna().pct_change().dropna()
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    er = r.rolling(window).mean()
    sd = r.rolling(window).std()
    return (er - rf_daily) / sd


def drawdown_series(series: pd.Series):
    s = series.dropna()
    peak = s.cummax()
    return (s / peak) - 1.0


def mcap_human(n):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    try:
        n = float(n)
    except Exception:
        return "—"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:,.0f}{unit}"
        n /= 1000.0
    return f"{n:,.0f}P"


def project_price_series(adj_close: pd.Series, horizon_days: int = 90, method: str = "CAGR"):
    """Proyección determinista de precio ajustado."""
    s = adj_close.dropna()
    if s.empty or len(s) < 5:
        return pd.DataFrame()
    last_date = s.index[-1]
    future_idx = pd.bdate_range(last_date + timedelta(days=1), periods=horizon_days)
    if method == "CAGR":
        n_days = (s.index[-1] - s.index[0]).days
        if n_days <= 0:
            return pd.DataFrame()
        total_return = s.iloc[-1] / s.iloc[0]
        daily_growth = total_return ** (1 / max(n_days, 1))
        proj = [s.iloc[-1] * (daily_growth ** (i+1)) for i in range(horizon_days)]
        return pd.DataFrame({"Projected": proj}, index=future_idx)
    y = np.log(s.values)
    x = np.arange(len(s))
    b1, b0 = np.polyfit(x, y, 1)
    y_pred = b0 + b1 * np.arange(len(s), len(s) + horizon_days)
    return pd.DataFrame({"Projected": np.exp(y_pred)}, index=future_idx)

# ---------- TRADUCCIÓN (cacheada) ----------
@st.cache_data(show_spinner=False)
def translate_to_spanish(text: str) -> str:
    """Traduce 'text' al español usando GoogleTranslator; en fallo devuelve original."""
    if not text:
        return text
    try:
        return GoogleTranslator(source="auto", target="es").translate(text)
    except Exception:
        return text

# ---------- MÉTRICAS MENSUALES (5 AÑOS) ----------
@st.cache_data(show_spinner=True)
def monthly_risk_return_last_5y(tickers, rf_annual: float):
    """
    Tabla mensual de los últimos 5 años:
      - Retorno mensual (%)
      - Volatilidad anualizada (std diaria del mes * sqrt(252))
      - Sharpe anualizado del mes (usando media diaria * 252)
    """
    end = date.today()
    start = end - timedelta(days=5*365 + 10)  # colchón
    rows = []
    for t in tickers:
        df = fetch_full_history(t, start=start, end=end, interval="1d")
        if df.empty or "Adj Close" not in df.columns:
            continue
        prices = df["Adj Close"].dropna()
        if prices.empty:
            continue
        r = prices.pct_change().dropna()
        by_month = r.groupby(pd.Grouper(freq="M"))
        for month_end, g in by_month:
            if g.empty:
                continue
            monthly_ret = (1.0 + g).prod() - 1.0
            ann_vol = g.std(ddof=0) * np.sqrt(252)
            ann_ret_from_mean = g.mean() * 252
            sharpe = (ann_ret_from_mean - rf_annual) / ann_vol if ann_vol and ann_vol > 0 else np.nan
            rows.append({
                "Ticker": t,
                "Mes": month_end.strftime("%Y-%m"),
                "Retorno mensual": monthly_ret,
                "Volatilidad anualizada": ann_vol,
                "Sharpe (anualizado)": sharpe
            })
    if not rows:
        return pd.DataFrame(columns=["Ticker", "Mes", "Retorno mensual", "Volatilidad anualizada", "Sharpe (anualizado)"])
    df_out = pd.DataFrame(rows).sort_values(["Ticker", "Mes"]).reset_index(drop=True)
    return df_out

# -------------------- IA: CLIENTE Y HELPERS --------------------
@st.cache_resource
def get_openai_client():
    """
    Devuelve un cliente de OpenAI usando la API key guardada en secrets.
    En Streamlit Cloud y en local se usa st.secrets["OPENAI_API_KEY"].
    """
    api_key = st.secrets.get("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)



def build_ai_context_for_tickers(tickers, dict_hist, rf):
    """
    Construye texto con info básica de cada ticker para dársela a la IA.
    Incluye: nombre, sector, país, retorno anualizado, volatilidad y Sharpe.
    """
    lines = []
    for t in tickers:
        df_t = dict_hist.get(t)
        if df_t is None or df_t.empty:
            continue
        info = fetch_static_info(t)
        long_name = info.get("long_name") or t
        sector = info.get("sector") or "Sin sector"
        country = info.get("country") or "—"
        series = df_t["Adj Close"]
        ann_ret, ann_vol, sharpe = annualized_metrics(series, rf_annual=rf)

        def fmt_pct(x):
            return "N/D" if pd.isna(x) else f"{x*100:.2f}%"

        def fmt_float(x):
            return "N/D" if pd.isna(x) else f"{x:.3f}"

        lines.append(
            f"Ticker: {t}\n"
            f"Nombre: {long_name}\n"
            f"Sector: {sector}\n"
            f"País: {country}\n"
            f"Retorno anualizado: {fmt_pct(ann_ret)}\n"
            f"Volatilidad anualizada: {fmt_pct(ann_vol)}\n"
            f"Sharpe: {fmt_float(sharpe)}\n"
            f"---"
        )
    return "\n".join(lines)


def generate_ai_overview_text(client: OpenAI, context_text: str) -> str:
    """Genera una explicación general de los tickers usando IA."""
    if not context_text:
        return "No hay suficiente información para analizar los tickers."
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un profesor de finanzas que explica portafolios a un estudiante "
                    "de 7mo semestre. Hablas en español, tono claro y didáctico. "
                    "No das consejos de compra/venta directos."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Te paso datos de varios tickers con su retorno, volatilidad y Sharpe. "
                    "Explica de forma sencilla el perfil de cada uno y comenta el conjunto en general.\n\n"
                    f"{context_text}"
                ),
            },
        ],
    )
    return completion.choices[0].message.content.strip()


def chat_with_ai_on_portfolio(client: OpenAI, context_text: str, history, question: str) -> str:
    """Responde preguntas del usuario usando el contexto del portafolio + historial."""
    base_messages = [
        {
            "role": "system",
            "content": (
                "Eres un profesor de finanzas que ayuda a un estudiante de 7mo semestre "
                "a entender un conjunto de acciones. Explicas en español, claro y sin "
                "recomendar comprar o vender nada."
            ),
        },
        {
            "role": "user",
            "content": (
                "Este es el contexto numérico de sus tickers. Úsalo como referencia para "
                "responder sus preguntas, pero no lo repitas completo salvo que sea necesario:\n\n"
                f"{context_text}"
            ),
        },
    ]
    messages = base_messages + history + [
        {"role": "user", "content": question}
    ]
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return completion.choices[0].message.content.strip()


def build_portfolio_stats_text(tickers, dict_hist, closes_df, rf_annual: float) -> str:
    """
    Construye un resumen de estadísticas de portafolio:
    - Retorno, volatilidad, Sharpe por ticker
    - Tickers más y menos arriesgados
    - Mejor y peor Sharpe
    - Resumen de correlaciones
    """
    lines = []
    lines.append(f"Número de tickers analizados: {len(tickers)}")

    metrics = []
    sector_counts = {}

    for t in tickers:
        df_t = dict_hist.get(t)
        if df_t is None or df_t.empty:
            continue

        info = fetch_static_info(t)
        sector = info.get("sector") or "Sin sector"
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

        series = df_t["Adj Close"]
        ann_ret, ann_vol, sharpe = annualized_metrics(series, rf_annual=rf_annual)
        metrics.append({
            "Ticker": t,
            "Sector": sector,
            "Retorno": ann_ret,
            "Volatilidad": ann_vol,
            "Sharpe": sharpe
        })

    if metrics:
        dfm = pd.DataFrame(metrics).set_index("Ticker")

        if sector_counts:
            sec_txt = ", ".join(f"{s}: {c}" for s, c in sector_counts.items())
            lines.append(f"Distribución por sectores (conteo de tickers): {sec_txt}")

        try:
            best_sharpe_t = dfm["Sharpe"].idxmax()
            best_sharpe_v = dfm["Sharpe"].max()
            lines.append(f"Mejor Sharpe: {best_sharpe_t} (≈ {best_sharpe_v:.3f})")
        except Exception:
            pass

        try:
            worst_sharpe_t = dfm["Sharpe"].idxmin()
            worst_sharpe_v = dfm["Sharpe"].min()
            lines.append(f"Peor Sharpe: {worst_sharpe_t} (≈ {worst_sharpe_v:.3f})")
        except Exception:
            pass

        try:
            most_vol_t = dfm["Volatilidad"].idxmax()
            most_vol_v = dfm["Volatilidad"].max()
            lines.append(f"Acción más volátil: {most_vol_t} (vol ≈ {most_vol_v*100:.2f}%)")
        except Exception:
            pass

        try:
            least_vol_t = dfm["Volatilidad"].idxmin()
            least_vol_v = dfm["Volatilidad"].min()
            lines.append(f"Acción menos volátil: {least_vol_t} (vol ≈ {least_vol_v*100:.2f}%)")
        except Exception:
            pass

    try:
        if closes_df is not None and not closes_df.empty and closes_df.shape[1] >= 2:
            rets = closes_df.pct_change().dropna()
            if not rets.empty:
                corr = rets.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                vals = corr.where(mask).stack()
                if not vals.empty:
                    mean_corr = vals.mean()
                    max_pair = vals.idxmax()
                    min_pair = vals.idxmin()
                    lines.append(
                        f"Correlación promedio entre pares de acciones: ≈ {mean_corr:.2f}"
                    )
                    lines.append(
                        f"Par más correlacionado: {max_pair[0]} – {max_pair[1]} (≈ {vals.max():.2f})"
                    )
                    lines.append(
                        f"Par menos correlacionado: {min_pair[0]} – {min_pair[1]} (≈ {vals.min():.2f})"
                    )
    except Exception:
        pass

    return "\n".join(lines)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yf_news_simple(ticker: str, max_items: int = 5):
    """
    Intenta obtener noticias desde yfinance.Ticker.news (si está disponible).
    Devuelve lista de dicts: {title, link, publisher}.
    """
    items = []
    try:
        tk = yf.Ticker(ticker)
        news = getattr(tk, "news", None)
        if isinstance(news, list):
            for n in news[:max_items]:
                title = n.get("title") or ""
                link = n.get("link") or n.get("url") or ""
                publisher = n.get("publisher") or ""
                if title:
                    items.append(
                        {"title": title, "link": link, "publisher": publisher}
                    )
    except Exception:
        pass
    return items


def tutor_explain_concept_ai(client: OpenAI, concept: str, context_text: str, stats_text: str) -> str:
    """Explica un concepto (Sharpe, volatilidad, etc.) usando los datos reales del portafolio."""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un profesor de finanzas explicando a un estudiante de 7mo semestre. "
                    "Usa ejemplos numéricos simples y, cuando puedas, conecta con los datos "
                    "del portafolio que te paso. Responde en español, tono claro y didáctico. "
                    "No des recomendaciones directas de compra/venta."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Quiero que me expliques el concepto de '{concept}' usando como contexto "
                    "este portafolio y sus estadísticas. Explica qué es, por qué importa y "
                    "cómo se ve reflejado en estos datos.\n\n"
                    f"DATOS POR TICKER:\n{context_text}\n\n"
                    f"RESUMEN DEL PORTAFOLIO:\n{stats_text}"
                ),
            },
        ],
    )
    return completion.choices[0].message.content.strip()


def tutor_generate_quiz_ai(client: OpenAI, context_text: str, stats_text: str):
    """
    Genera una pregunta tipo quiz (opción múltiple) en formato JSON:
    {
      "pregunta": "...",
      "opciones": ["A) ...", "B) ...", ...],
      "respuesta_correcta": "A) ...",
      "explicacion": "..."
    }
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un profesor de finanzas que diseña un pequeño quiz para un estudiante. "
                    "Usa el contexto del portafolio solo como inspiración para el tipo de pregunta "
                    "(Sharpe, volatilidad, correlación, diversificación, etc.). "
                    "NO incluyas recomendaciones de inversión."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Crea UNA sola pregunta de opción múltiple (3 o 4 opciones) sobre conceptos de "
                    "finanzas de portafolio (Sharpe, volatilidad, diversificación, correlación, etc.). "
                    "La pregunta debe ser de nivel 7mo semestre pero entendible.\n\n"
                    "Devuelve la respuesta EN FORMATO JSON con las claves EXACTAS:\n"
                    " - 'pregunta' (string)\n"
                    " - 'opciones' (lista de strings)\n"
                    " - 'respuesta_correcta' (string, exactamente igual a una de las opciones)\n"
                    " - 'explicacion' (string, explicación breve en español)\n\n"
                    "No incluyas nada más fuera del JSON.\n\n"
                    f"Contexto numérico de ejemplo:\n{context_text}\n\n"
                    f"Resumen del portafolio:\n{stats_text}"
                ),
            },
        ],
    )
    raw = completion.choices[0].message.content
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        if "pregunta" not in data or "opciones" not in data or "respuesta_correcta" not in data:
            return None
        if not isinstance(data["opciones"], list) or len(data["opciones"]) == 0:
            return None
        return data
    except Exception:
        return None


def scanner_ai_suggestions(client: OpenAI, df_metrics: pd.DataFrame, preference_text: str) -> str:
    """
    Usa IA para analizar las oportunidades dentro de los tickers cargados,
    según una descripción de lo que el usuario busca.
    """
    if df_metrics is None or df_metrics.empty:
        return "No hay métricas suficientes para analizar oportunidades."

    df_to_send = df_metrics.copy()
    cols_keep = [c for c in ["Ticker", "Sector", "Retorno", "Volatilidad", "Sharpe"] if c in df_to_send.columns]
    df_to_send = df_to_send[cols_keep]

    table_str = df_to_send.to_string(index=False)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un analista de portafolios. Recibirás una tabla con acciones y sus "
                    "métricas (retorno, volatilidad, Sharpe, sector) y una descripción del tipo "
                    "de oportunidad que busca el usuario. Debes:\n"
                    "1) Explicar qué tipo de perfil está describiendo el usuario.\n"
                    "2) Elegir algunas acciones (entre los tickers de la tabla) que parezcan más "
                    "alineadas con ese perfil.\n"
                    "3) Justificar tu elección usando únicamente esas métricas (retorno, riesgo, Sharpe, sector).\n"
                    "4) No des recomendaciones explícitas de compra/venta, solo análisis y orden de preferencia.\n"
                    "Responde en español, con buena estructura, bullets o numeración."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Descripción del tipo de oportunidad que busco:\n{preference_text}\n\n"
                    f"Esta es la tabla de acciones disponibles y sus métricas:\n{table_str}\n\n"
                    "Analiza cuáles parecen más alineadas con lo que busco y explícalo."
                ),
            },
        ],
    )
    return completion.choices[0].message.content.strip()

# -------------------- ACCIONES POR PAÍS (LISTAS PREDEFINIDAS) --------------------
COUNTRY_TICKERS = {
    "Estados Unidos": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
        "V", "PG", "XOM", "UNH", "HD", "MA", "BAC", "PFE", "ABBV", "KO",
        "PEP", "TMO", "AVGO", "ADBE", "CSCO", "CMCSA", "NFLX", "COST", "NKE", "WMT",
        "DIS", "INTC", "AMD", "QCOM", "TXN", "MRK", "ABT", "CRM", "MCD", "ACN",
        "DHR", "IBM", "AMGN", "HON", "ORCL", "LIN", "SBUX", "LOW", "GE", "CAT"
    ],

    "México": [
        "AMXL.MX", "WALMEX.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "CEMEXCPO.MX",
        "TLEVISACPO.MX", "ALSEA.MX", "GRUMAB.MX", "BIMBOA.MX", "KOFUBL.MX",
        "GAPB.MX", "ASURB.MX", "OMAB.MX", "BBAJIOO.MX", "LIVEPOLC1.MX",
        "BOLSAA.MX", "GFINBURO.MX", "GENTERA.MX", "IENOVA.MX", "AC.MX"
    ],

    "Canadá": [
        "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO",
        "NA.TO", "BCE.TO", "T.TO", "ENB.TO", "TRP.TO",
        "SU.TO", "CNQ.TO", "SHOP.TO", "NTR.TO", "MFC.TO",
        "SLF.TO", "CNR.TO", "CP.TO", "BAM.TO", "FTS.TO",
        "EMA.TO", "IFC.TO", "WCN.TO", "MG.TO", "DOL.TO"
    ],

    "España": [
        "SAN.MC", "BBVA.MC", "IBE.MC", "ITX.MC", "REP.MC",
        "TEF.MC", "FER.MC", "ACS.MC", "AENA.MC", "IAG.MC",
        "MTS.MC", "GRF.MC", "CLNX.MC", "ENG.MC", "RED.MC",
        "MAP.MC"
    ],

    "Alemania": [
        "SAP.DE", "DTE.DE", "ALV.DE", "SIE.DE", "BMW.DE",
        "VOW3.DE", "BAS.DE", "BAYN.DE", "MUV2.DE", "LIN.DE",
        "ADS.DE", "DPW.DE", "HEN3.DE", "IFX.DE", "RWE.DE",
        "EOAN.DE", "DBK.DE", "FME.DE", "FRE.DE", "MRK.DE"
    ],

    "Reino Unido": [
        "HSBA.L", "BP.L", "GSK.L", "ULVR.L", "AZN.L",
        "SHEL.L", "RIO.L", "BATS.L", "DGE.L", "LLOY.L",
        "BARC.L", "VOD.L", "NG.L", "BA.L", "REL.L",
        "TSCO.L", "BT-A.L", "RR.L", "SMT.L", "GLEN.L"
    ]
}

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Parámetros")

suggested = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BRK-B", "KO"]

tickers_input = st.sidebar.text_input("Tickers (separados por coma)", value="AAPL, MSFT, NVDA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
st.sidebar.caption("Sugeridos: " + ", ".join(suggested))

today = date.today()
default_start = today - timedelta(days=365)
start_date = st.sidebar.date_input("Fecha inicio", value=default_start, max_value=today)
end_date = st.sidebar.date_input("Fecha fin", value=today, max_value=today)
if start_date > end_date:
    st.sidebar.error("La fecha de inicio no puede ser mayor a la de fin.")

interval = st.sidebar.selectbox("Intervalo", options=["1d", "1wk", "1mo"], index=0, help="Intervalo de las velas/muestras")

# Índices para comparar
st.sidebar.markdown("### Índices de referencia")
index_options = {"^GSPC": "S&P 500", "^NDX": "Nasdaq 100", "^DJI": "Dow Jones", "^IXIC": "Nasdaq Comp", "^MXX": "IPC México"}
selected_indices = st.sidebar.multiselect("Selecciona índices", options=list(index_options.keys()), default=["^GSPC", "^NDX"])

# Parámetros de riesgo
rf_pct = st.sidebar.number_input("Tasa libre anual (%)", value=5.5, step=0.1, help="Afecta el Sharpe")
rf = rf_pct / 100.0
roll_vol_win = st.sidebar.slider("Ventana vol. rodante (días)", 10, 126, 21, step=1)
roll_sharpe_win = st.sidebar.slider("Ventana Sharpe rodante (días)", 21, 252, 63, step=1)

# Proyección (solo si 1 ticker)
st.sidebar.markdown("### Proyección (1 ticker)")
proj_days = st.sidebar.slider("Horizonte proyección (días hábiles)", 10, 252, 90, step=5)
proj_method = st.sidebar.selectbox("Método de proyección", ["CAGR", "Regresión"], index=0)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Actualizar datos")

# -------------------- MAIN --------------------
st.title("💼 APP Finanzas - Bernardo Sánchez")
st.markdown(
    '<p style="color:#6b7280; font-size:0.95rem; margin-top:-0.8rem;">'
    'Dashboard interactivo para análisis de acciones, portafolios e IA financiera.'
    '</p>',
    unsafe_allow_html=True
)

if not tickers:
    st.info("Ingresa al menos un ticker (ej. **AAPL, MSFT**).")
    st.stop()

# Forzar recarga al pulsar botón
if "last_run" not in st.session_state or run_btn:
    st.session_state["last_run"] = datetime.now()

# 1) Datos de tus tickers (rango seleccionado)
dict_hist, closes_df = fetch_history(tickers, start_date, end_date, interval=interval)
if closes_df.empty:
    st.warning("No se obtuvieron datos. Verifica tickers, fechas o intervalo.")
    st.stop()

# 2) Datos de índices (para comparativa en rango)
idx_hist, idx_closes = ({}, pd.DataFrame())
if selected_indices:
    idx_hist, idx_closes = fetch_history(selected_indices, start_date, end_date, interval=interval)

# -------------------- TABS PRINCIPALES --------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Análisis",
    "🧮 Simulador de Portafolio",
    "🌍 Acciones por país",
    "🧠 Análisis con IA",
    "🎓 Tutor de Finanzas (IA)",
    "📰 Sentimiento & Noticias (IA)",
    "🔍 Scanner de oportunidades (IA)"
])

# =========================================================
# TAB 1: ANÁLISIS
# =========================================================
with tab1:
    st.subheader("Información general")
    cols = st.columns(min(4, max(1, len(tickers))))
    for i, t in enumerate(tickers[:12]):
        df_t = dict_hist.get(t)
        info = fetch_static_info(t)
        price, change, last_dt = latest_price_and_change(df_t)
        logo_img = fetch_logo_image(info, size=40)

        with cols[i % len(cols)]:
            head1, head2 = st.columns([1, 5])
            with head1:
                if logo_img:
                    st.image(logo_img, width=40)
                else:
                    st.markdown("🧩")
            with head2:
                name = info.get('long_name') or t
                today_price, today_chg = fetch_today_quote(t)
                if today_price is not None:
                    chg_sign = "+" if (today_chg is not None and today_chg >= 0) else ""
                    chg_txt = f"{chg_sign}{today_chg:.2f}%" if today_chg is not None else "—"
                    st.markdown(
                        f"""**{name}**&nbsp;&nbsp;
                        <span style="font-size:0.9rem;color:#6b7280;">${today_price:,.2f} ({chg_txt} hoy)</span><br>
                        <code>{t}</code>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"**{name}**  \n`{t}`")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Precio (fin de rango)", "—" if price is None or np.isnan(price) else f"${price:,.2f}")
            with c2:
                delta_txt = "—" if change is None or np.isnan(change) else f"{change:+.2f}%"
                st.metric("Cambio vs periodo previo", delta_txt)

            when_txt = f"{last_dt.date()}" if last_dt is not None else "—"
            st.caption(
                f"**Último dato:** {when_txt} | "
                f"**Mcap:** {mcap_human(info.get('market_cap'))} | "
                f"**Sector:** {info.get('sector') or '—'} | "
                f"**País:** {info.get('country') or '—'}"
            )

    with st.expander("📄 Descripción de las empresas"):
        for t in tickers:
            info = fetch_static_info(t)
            name = info.get("long_name") or t
            st.markdown(f"**{name}** (`{t}`)")
            summary = info.get("summary")
            summary_es = translate_to_spanish(summary) if summary else None
            st.write(summary_es or "*Sin descripción disponible*")
            st.markdown("---")

    st.subheader("Gráficas de velas (OHLC) — una por ticker")
    cols_per_row = 2
    for i, t in enumerate(tickers):
        if t in dict_hist and not dict_hist[t].empty:
            df = dict_hist[t]
            ohlc = df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
            if ohlc.empty:
                continue
            if i % cols_per_row == 0:
                row = st.columns(cols_per_row)
            col = row[i % cols_per_row]
            with col:
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=ohlc.index,
                            open=ohlc["Open"],
                            high=ohlc["High"],
                            low=ohlc["Low"],
                            close=ohlc["Close"],
                            name=t
                        )
                    ]
                )
                fig.update_layout(
                    title=f"Velas — {t}",
                    xaxis_title="Fecha",
                    yaxis_title="Precio",
                    height=520,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparativa: precios ajustados")
    fig_line = go.Figure()
    for t in tickers:
        if t in dict_hist and not dict_hist[t].empty:
            fig_line.add_trace(go.Scatter(x=dict_hist[t].index, y=dict_hist[t]["Adj Close"], mode="lines", name=t))
    fig_line.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Adj Close",
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Precio ajustado + proyección")
    if len(tickers) == 1 and tickers[0] in dict_hist:
        t = tickers[0]
        df = dict_hist[t]
        base = df["Adj Close"].dropna()
        if not base.empty:
            proj_df = project_price_series(base, horizon_days=proj_days, method=proj_method)
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(x=base.index, y=base.values, mode="lines", name=f"{t} Adj Close"))
            if not proj_df.empty:
                fig_proj.add_trace(
                    go.Scatter(
                        x=proj_df.index, y=proj_df["Projected"], mode="lines",
                        name=f"Proyección ({proj_method})", line=dict(dash="dash")
                    )
                )
            fig_proj.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Adj Close",
                height=460,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_proj, use_container_width=True)
        else:
            st.info("Sin datos suficientes para proyectar.")
    else:
        st.info("Selecciona **un solo ticker** para ver la proyección de precio.")

    st.subheader("Comparativa vs índices (base=100)")
    if not closes_df.empty:
        comp = {}
        for t in tickers:
            if t in dict_hist and not dict_hist[t].empty:
                s = dict_hist[t]["Adj Close"].dropna()
                if not s.empty:
                    comp[t] = 100 * s / s.iloc[0]
        if not idx_closes.empty:
            for idx in idx_closes.columns:
                s = idx_closes[idx].dropna()
                if not s.empty:
                    comp[index_options.get(idx, idx)] = 100 * s / s.iloc[0]
        if comp:
            df_norm = pd.DataFrame(comp).dropna(how="all")
            fig_cmp = go.Figure()
            for col in df_norm.columns:
                fig_cmp.add_trace(go.Scatter(x=df_norm.index, y=df_norm[col], mode="lines", name=col))
            fig_cmp.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Índice (base=100)",
                height=500,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info("No hay series suficientes para la comparativa normalizada.")
    else:
        st.info("No hay datos para la comparativa vs índices.")

    st.subheader("Riesgo y rendimiento (anualizados)")
    metrics_rows = []
    mcaps = {}
    for t in tickers:
        info = fetch_static_info(t)
        mcaps[t] = info.get("market_cap")
        if t in dict_hist and not dict_hist[t].empty:
            series = dict_hist[t]["Adj Close"]
            ann_ret, ann_vol, sharpe = annualized_metrics(series, rf_annual=rf)
            metrics_rows.append({
                "Ticker": t,
                "Retorno anualizado": ann_ret,
                "Volatilidad anualizada": ann_vol,
                "Sharpe": sharpe,
                "Mcap": mcaps[t]
            })

    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows).set_index("Ticker")
        fmt = {
            "Retorno anualizado": lambda x: "—" if pd.isna(x) else f"{x*100:,.2f}%",
            "Volatilidad anualizada": lambda x: "—" if pd.isna(x) else f"{x*100:,.2f}%",
            "Sharpe": lambda x: "—" if pd.isna(x) else f"{x:,.3f}",
            "Mcap": mcap_human
        }
        st.dataframe(df_metrics.style.format(fmt), use_container_width=True, height=240)

        st.markdown("##### Dispersión riesgo-rendimiento (burbujas)")
        tmp = df_metrics.reset_index().rename(columns={"Volatilidad anualizada": "Vol", "Retorno anualizado": "Return"})
        tmp["Mcap_num"] = tmp["Mcap"].apply(lambda x: np.nan if x is None else float(x))
        fig_bub = px.scatter(
            tmp,
            x="Vol", y="Return",
            size="Mcap_num", color="Sharpe", hover_name="Ticker",
            size_max=60,
            labels={"Vol": "Volatilidad", "Return": "Retorno"},
            title="Riesgo vs Rendimiento (rango seleccionado)"
        )
        fig_bub.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bub, use_container_width=True)
    else:
        st.info("No hay suficientes datos para calcular métricas.")

    st.subheader("Correlaciones de retornos")
    rets = closes_df.pct_change().dropna()
    if not rets.empty and rets.shape[1] >= 2:
        corr = rets.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Heatmap de correlación (retornos)"
        )
        fig_corr.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.download_button(
            "⬇️ Descargar correlaciones (CSV)",
            corr.to_csv().encode("utf-8"),
            file_name="correlaciones.csv",
            mime="text/csv"
        )
    else:
        st.info("Se necesitan al menos 2 tickers para correlación.")

    st.subheader("Riesgo dinámico: volatilidad y Sharpe rodante")
    if not closes_df.empty:
        tabs2 = st.tabs(["Volatilidad rodante", "Sharpe rodante"])
        with tabs2[0]:
            fig_rv = go.Figure()
            for t in tickers:
                if t in dict_hist and not dict_hist[t].empty:
                    rv = rolling_vol(dict_hist[t]["Adj Close"], window=roll_vol_win)
                    if not rv.empty:
                        fig_rv.add_trace(go.Scatter(x=rv.index, y=rv, mode="lines", name=t))
            fig_rv.update_layout(
                title=f"Volatilidad anualizada rodante (ventana={roll_vol_win}d)",
                xaxis_title="Fecha", yaxis_title="Volatilidad",
                height=460, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_rv, use_container_width=True)

        with tabs2[1]:
            fig_rs = go.Figure()
            for t in tickers:
                if t in dict_hist and not dict_hist[t].empty:
                    rs = rolling_sharpe(dict_hist[t]["Adj Close"], rf_annual=rf, window=roll_sharpe_win)
                    if not rs.empty:
                        fig_rs.add_trace(go.Scatter(x=rs.index, y=rs, mode="lines", name=t))
            fig_rs.update_layout(
                title=f"Sharpe rodante (ventana={roll_sharpe_win}d)",
                xaxis_title="Fecha", yaxis_title="Sharpe (ventana)",
                height=460, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_rs, use_container_width=True)

    st.subheader("Descargar datos")
    if not closes_df.empty:
        csv_bytes = closes_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar Adj Close (CSV)",
            data=csv_bytes,
            file_name=f"adj_close_{start_date}_{end_date}.csv",
            mime="text/csv"
        )

    long_frames = []
    for t, df in dict_hist.items():
        tmp = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
        tmp["Ticker"] = t
        long_frames.append(tmp)
    if long_frames:
        long_df = pd.concat(long_frames, axis=0, ignore_index=True)
        csv_long = long_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar OHLCV (CSV, formato largo)",
            data=csv_long,
            file_name=f"ohlcv_{start_date}_{end_date}.csv",
            mime="text/csv"
        )

    st.subheader("Tabla mensual (últimos 5 años): Riesgo vs Rendimiento")
    monthly_df = monthly_risk_return_last_5y(tickers, rf_annual=rf)

    if not monthly_df.empty:
        fmt_month = {
            "Retorno mensual": lambda x: "—" if pd.isna(x) else f"{x*100:,.2f}%",
            "Volatilidad anualizada": lambda x: "—" if pd.isna(x) else f"{x*100:,.2f}%",
            "Sharpe (anualizado)": lambda x: "—" if pd.isna(x) else f"{x:,.3f}",
        }
        st.dataframe(monthly_df.style.format(fmt_month), use_container_width=True, height=420)
        csv_monthly = monthly_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar tabla mensual (5 años) Riesgo vs Rendimiento (CSV)",
            data=csv_monthly,
            file_name="riesgo_rendimiento_mensual_5y.csv",
            mime="text/csv"
        )
    else:
        st.info("No fue posible construir la tabla mensual de 5 años. Verifica los tickers.")

    st.caption("Fuente: Yahoo Finance (yfinance) • Logos: Yahoo / Clearbit (si disponible).")

# =========================================================
# TAB 2: SIMULADOR DE PORTAFOLIO
# =========================================================
with tab2:
    st.subheader("Simulador de Portafolio (monto, riesgo y sector)")

    if closes_df.empty:
        st.info("Primero selecciona tickers válidos en la barra lateral.")
    else:
        monto = st.number_input("Monto total a invertir (USD)", min_value=0.0, value=10000.0, step=1000.0)
        nivel_riesgo = st.slider(
            "Nivel de riesgo deseado",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = muy conservador, 5 = muy agresivo"
        )
        n_max_tickers = st.slider(
            "Número máximo de acciones en el portafolio",
            min_value=1,
            max_value=min(10, len(tickers)),
            value=min(5, len(tickers))
        )

        metrics_rows_sim = []
        for t in tickers:
            if t in dict_hist and not dict_hist[t].empty:
                info = fetch_static_info(t)
                sector = info.get("sector") or "Sin sector"
                series = dict_hist[t]["Adj Close"]
                ann_ret, ann_vol, sharpe = annualized_metrics(series, rf_annual=rf)
                metrics_rows_sim.append({
                    "Ticker": t,
                    "Sector": sector,
                    "Retorno": ann_ret,
                    "Volatilidad": ann_vol,
                    "Sharpe": sharpe
                })

        if not metrics_rows_sim:
            st.info("No hay suficientes datos para construir el simulador.")
        else:
            df_sim = pd.DataFrame(metrics_rows_sim)
            sectores_disponibles = sorted(df_sim["Sector"].dropna().unique().tolist())
            sectores_sel = st.multiselect(
                "Sectores deseados",
                options=sectores_disponibles,
                default=sectores_disponibles,
                help="Filtra el universo de acciones por sector."
            )

            df_cand = df_sim.copy()
            if sectores_sel:
                df_cand = df_cand[df_cand["Sector"].isin(sectores_sel)]

            if df_cand.empty:
                st.warning("No hay acciones en los sectores seleccionados.")
            else:
                df_cand = df_cand.dropna(subset=["Retorno", "Volatilidad", "Sharpe"], how="all").copy()
                if df_cand.empty:
                    st.info("No hay suficientes métricas para las acciones filtradas.")
                else:
                    if nivel_riesgo <= 2:
                        df_cand = df_cand.dropna(subset=["Volatilidad"])
                        df_cand = df_cand.sort_values("Volatilidad", ascending=True)
                        df_cand = df_cand.head(n_max_tickers)
                        df_cand["score"] = 1.0 / df_cand["Volatilidad"].replace(0, np.nan)
                    elif nivel_riesgo == 3:
                        df_cand = df_cand.dropna(subset=["Sharpe"])
                        df_cand = df_cand.sort_values("Sharpe", ascending=False)
                        df_cand = df_cand.head(n_max_tickers)
                        df_cand["score"] = df_cand["Sharpe"].clip(lower=0)
                    else:
                        df_cand = df_cand.dropna(subset=["Retorno"])
                        df_cand = df_cand.sort_values("Retorno", ascending=False)
                        df_cand = df_cand.head(n_max_tickers)
                        df_cand["score"] = df_cand["Retorno"].clip(lower=0)

                    if df_cand.empty:
                        st.info("No fue posible seleccionar acciones con las condiciones dadas.")
                    else:
                        df_cand["score"].replace([np.inf, -np.inf], np.nan, inplace=True)
                        df_cand = df_cand.dropna(subset=["score"])
                        if df_cand.empty:
                            st.info("Las métricas no permiten construir un portafolio válido.")
                        else:
                            total_score = df_cand["score"].sum()
                            if total_score <= 0:
                                pesos = np.ones(len(df_cand)) / len(df_cand)
                            else:
                                pesos = df_cand["score"] / total_score
                            df_cand["Peso sugerido"] = pesos

                            precios = []
                            for t in df_cand["Ticker"]:
                                p_hoy, _ = fetch_today_quote(t)
                                if p_hoy is None and t in dict_hist and not dict_hist[t].empty:
                                    p_hoy = float(dict_hist[t]["Adj Close"].dropna().iloc[-1])
                                precios.append(p_hoy if p_hoy is not None else np.nan)
                            df_cand["Precio actual"] = precios

                            filas_port = []
                            invertido_total = 0.0
                            for _, row in df_cand.iterrows():
                                p = row["Precio actual"]
                                w = row["Peso sugerido"]
                                if pd.isna(p) or p <= 0 or monto <= 0:
                                    n_shares = 0
                                    invertido = 0.0
                                else:
                                    asignado = monto * w
                                    n_shares = int(np.floor(asignado / p))
                                    invertido = n_shares * p
                                invertido_total += invertido
                                filas_port.append({
                                    "Ticker": row["Ticker"],
                                    "Sector": row["Sector"],
                                    "Precio actual": p,
                                    "Peso sugerido": w,
                                    "Acciones": n_shares,
                                    "Invertido": invertido
                                })

                            if invertido_total <= 0:
                                st.warning("Con el monto y precios actuales no se pudo comprar ni una acción. Prueba aumentando el monto.")
                            else:
                                df_port = pd.DataFrame(filas_port)
                                df_port["% del portafolio"] = df_port["Invertido"] / invertido_total
                                efectivo = monto - invertido_total

                                st.markdown("### Sugerencia de portafolio")
                                st.dataframe(
                                    df_port.assign(
                                        **{
                                            "Precio actual": df_port["Precio actual"].map(lambda x: "—" if pd.isna(x) else f"${x:,.2f}"),
                                            "Peso sugerido": df_port["Peso sugerido"].map(lambda x: f"{x*100:,.1f}%"),
                                            "Invertido": df_port["Invertido"].map(lambda x: f"${x:,.2f}"),
                                            "% del portafolio": df_port["% del portafolio"].map(lambda x: f"{x*100:,.1f}%")
                                        }
                                    ),
                                    use_container_width=True,
                                    height=320
                                )

                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Monto total", f"${monto:,.2f}")
                                with col_b:
                                    st.metric("Total invertido", f"${invertido_total:,.2f}")
                                with col_c:
                                    st.metric("Efectivo restante", f"${efectivo:,.2f}")

                                tickers_sel = df_port["Ticker"].tolist()
                                rets_sel = closes_df[tickers_sel].pct_change().dropna()
                                port_ret_ann = np.nan
                                port_vol_ann = np.nan
                                port_sharpe = np.nan
                                if not rets_sel.empty:
                                    mean_rets_daily = rets_sel.mean()
                                    cov_daily = rets_sel.cov()
                                    w_vec = df_port["Peso sugerido"].values
                                    port_ret_ann = (1 + mean_rets_daily.dot(w_vec))**252 - 1
                                    port_vol_ann = float(np.sqrt(np.dot(w_vec.T, np.dot(cov_daily * 252, w_vec))))
                                    if port_vol_ann > 0:
                                        port_sharpe = (port_ret_ann - rf) / port_vol_ann

                                st.markdown("### Métricas del portafolio sugerido")
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.metric(
                                        "Retorno anual esperado",
                                        "—" if pd.isna(port_ret_ann) else f"{port_ret_ann*100:,.2f}%"
                                    )
                                with c2:
                                    st.metric(
                                        "Volatilidad anual esperada",
                                        "—" if pd.isna(port_vol_ann) else f"{port_vol_ann*100:,.2f}%"
                                    )
                                with c3:
                                    st.metric(
                                        "Sharpe esperado",
                                        "—" if pd.isna(port_sharpe) else f"{port_sharpe:,.3f}"
                                    )

                                st.markdown("### Distribución de pesos")
                                fig_pie_t = px.pie(
                                    df_port,
                                    names="Ticker",
                                    values="% del portafolio",
                                    title="Pesos por ticker"
                                )
                                st.plotly_chart(fig_pie_t, use_container_width=True)

                                fig_pie_s = px.pie(
                                    df_port.groupby("Sector", as_index=False)["Invertido"].sum(),
                                    names="Sector",
                                    values="Invertido",
                                    title="Pesos por sector (según monto invertido)"
                                )
                                st.plotly_chart(fig_pie_s, use_container_width=True)

# =========================================================
# TAB 3: ACCIONES POR PAÍS
# =========================================================
with tab3:
    st.subheader("🌍 Acciones por país (Top acciones por mercado)")

    country = st.selectbox(
        "Selecciona el país",
        options=list(COUNTRY_TICKERS.keys()),
        index=0
    )

    tickers_country = COUNTRY_TICKERS.get(country, [])

    if not tickers_country:
        st.info("No hay tickers configurados para este país.")
    else:
        st.caption(
            "Lista predefinida de acciones representativas de cada país. "
            "Se muestra información actual usando Yahoo Finance."
        )

        max_n = min(50, len(tickers_country))
        top_n = st.slider(
            "¿Cuántas acciones quieres ver?",
            min_value=5,
            max_value=max_n,
            value=max_n,
            step=5
        )

        subset = tickers_country[:top_n]

        rows_table = []
        for t in subset:
            info = fetch_static_info(t)
            today_price, today_chg = fetch_today_quote(t)

            name = info.get("long_name") or t
            sector = info.get("sector") or "Sin sector"
            country_info = info.get("country") or "—"
            mcap = info.get("market_cap")

            rows_table.append({
                "Ticker": t,
                "Nombre": name,
                "País (info Yahoo)": country_info,
                "Sector": sector,
                "Precio actual": np.nan if today_price is None else today_price,
                "Cambio hoy %": np.nan if today_chg is None else today_chg,
                "Market Cap": mcap
            })

        if rows_table:
            df_country = pd.DataFrame(rows_table)

            st.markdown("### Tabla resumida (ordenable)")
            fmt_country = {
                "Precio actual": lambda x: "—" if pd.isna(x) else f"${x:,.2f}",
                "Cambio hoy %": lambda x: "—" if pd.isna(x) else f"{x:,.2f}%",
                "Market Cap": mcap_human
            }
            st.dataframe(
                df_country.style.format(fmt_country),
                use_container_width=True,
                height=400
            )

            st.markdown("### Vista detallada con logos")

            for t in subset:
                info = fetch_static_info(t)
                logo_img = fetch_logo_image(info, size=40)
                today_price, today_chg = fetch_today_quote(t)

                name = info.get("long_name") or t
                sector = info.get("sector") or "Sin sector"
                country_info = info.get("country") or "—"
                mcap = info.get("market_cap")
                website = info.get("website")

                cambio_txt = "—"
                if today_chg is not None and not pd.isna(today_chg):
                    cambio_txt = f"{today_chg:+.2f}%"

                precio_txt = "—" if today_price is None or pd.isna(today_price) else f"${today_price:,.2f}"

                with st.container():
                    c1, c2, c3, c4 = st.columns([1, 3, 2, 2])

                    with c1:
                        if logo_img:
                            st.image(logo_img, width=40)
                        else:
                            st.markdown("🧩")

                    with c2:
                        st.markdown(f"**{name}**  \n`{t}`")
                        st.caption(f"Sector: {sector} | País (Yahoo): {country_info}")
                        if website:
                            st.markdown(f"[Sitio oficial]({website})")

                    with c3:
                        st.metric("Precio actual", precio_txt, delta=cambio_txt)

                    with c4:
                        st.metric("Market Cap", mcap_human(mcap))

                    st.markdown("---")
        else:
            st.info("No se pudo construir la tabla de este país. Intenta con otro o revisa tu conexión.")

# =========================================================
# TAB 4: ANÁLISIS CON IA
# =========================================================
with tab4:
    st.subheader("Análisis con IA de tus tickers")

    client = get_openai_client()
    if client is None:
        st.warning(
            "Configura tu API key dentro de la función get_openai_client() "
            "(línea donde dice TU_API_KEY_AQUI)."
        )
    else:
        context_text = build_ai_context_for_tickers(tickers, dict_hist, rf)
        if not context_text:
            st.info("No hay información suficiente de los tickers para construir el contexto de IA.")
        else:
            ai_key = f"{sorted(tickers)}|{start_date}|{end_date}|{interval}|{rf}"
            if (
                "ai_overview_key" not in st.session_state
                or st.session_state.get("ai_overview_key") != ai_key
            ):
                try:
                    overview_text = generate_ai_overview_text(client, context_text)
                except Exception as e:
                    overview_text = f"No se pudo generar el análisis con IA. Error: {e}"
                st.session_state["ai_overview_key"] = ai_key
                st.session_state["ai_overview_text"] = overview_text

            st.markdown("### Explicación general de tus tickers (IA)")
            st.write(st.session_state.get("ai_overview_text", ""))

            st.markdown("---")
            st.markdown("### Chat con IA sobre tu portafolio")

            if "ai_chat_history" not in st.session_state:
                st.session_state["ai_chat_history"] = []

            user_question = st.text_input(
                "Haz una pregunta a la IA sobre estos tickers (riesgo, comparación, diversificación, etc.)"
            )
            ask_btn = st.button("Preguntar a la IA")

            if ask_btn and user_question.strip():
                try:
                    answer = chat_with_ai_on_portfolio(
                        client,
                        context_text=context_text,
                        history=st.session_state["ai_chat_history"],
                        question=user_question.strip(),
                    )
                except Exception as e:
                    answer = f"No se pudo obtener respuesta de la IA. Error: {e}"

                st.session_state["ai_chat_history"].append(
                    {"role": "user", "content": user_question.strip()}
                )
                st.session_state["ai_chat_history"].append(
                    {"role": "assistant", "content": answer}
                )

            if st.session_state["ai_chat_history"]:
                st.markdown("#### Historial")
                for msg in st.session_state["ai_chat_history"]:
                    if msg["role"] == "user":
                        st.markdown(f"**Tú:** {msg['content']}")
                    else:
                        st.markdown(f"**IA:** {msg['content']}")

# =========================================================
# TAB 5: TUTOR DE FINANZAS (IA)
# =========================================================
with tab5:
    st.subheader("🎓 Tutor de Finanzas (IA)")

    client = get_openai_client()
    if client is None:
        st.warning(
            "Configura tu API key dentro de la función get_openai_client() "
            "para usar el tutor de finanzas."
        )
    else:
        base_context = build_ai_context_for_tickers(tickers, dict_hist, rf)
        stats_text = build_portfolio_stats_text(tickers, dict_hist, closes_df, rf)
        full_context = base_context + "\n\n" + stats_text

        st.markdown("#### Explicador de conceptos con tus propios datos")

        concepto = st.selectbox(
            "Elige un concepto que quieras entender mejor:",
            [
                "Retorno anualizado",
                "Volatilidad",
                "Ratio de Sharpe",
                "Correlación entre acciones",
                "Diversificación de portafolio",
                "Drawdown (caída desde máximos)"
            ]
        )

        if st.button("📘 Explicar concepto usando mi portafolio"):
            try:
                explicacion = tutor_explain_concept_ai(
                    client,
                    concepto,
                    base_context,
                    stats_text
                )
            except Exception as e:
                explicacion = f"No se pudo generar la explicación. Error: {e}"

            st.markdown("**Explicación del tutor:**")
            st.write(explicacion)

        st.markdown("---")
        st.markdown("#### Pregunta libre al tutor de finanzas")

        if "tutor_chat_history" not in st.session_state:
            st.session_state["tutor_chat_history"] = []

        pregunta_tutor = st.text_input(
            "Escribe tu pregunta (por ejemplo: "
            "'¿por qué un Sharpe alto es bueno?' o "
            "'¿qué significa que dos acciones tengan alta correlación?')"
        )
        if st.button("❓ Preguntar al tutor"):
            if pregunta_tutor.strip():
                try:
                    respuesta = chat_with_ai_on_portfolio(
                        client,
                        context_text=full_context,
                        history=st.session_state["tutor_chat_history"],
                        question=pregunta_tutor.strip()
                    )
                except Exception as e:
                    respuesta = f"No se pudo obtener respuesta de la IA. Error: {e}"

                st.session_state["tutor_chat_history"].append(
                    {"role": "user", "content": pregunta_tutor.strip()}
                )
                st.session_state["tutor_chat_history"].append(
                    {"role": "assistant", "content": respuesta}
                )

        if st.session_state.get("tutor_chat_history"):
            st.markdown("#### Historial con el tutor")
            for msg in st.session_state["tutor_chat_history"]:
                if msg["role"] == "user":
                    st.markdown(f"**Tú:** {msg['content']}")
                else:
                    st.markdown(f"**Tutor (IA):** {msg['content']}")

        st.markdown("---")
        st.markdown("#### Quiz rápido (conceptos de tu portafolio)")

        if "tutor_quiz" not in st.session_state:
            st.session_state["tutor_quiz"] = None
        if "tutor_quiz_feedback" not in st.session_state:
            st.session_state["tutor_quiz_feedback"] = ""

        if st.button("🧪 Generar pregunta de quiz"):
            try:
                quiz_data = tutor_generate_quiz_ai(client, base_context, stats_text)
            except Exception as e:
                quiz_data = None
                st.error(f"No se pudo generar la pregunta. Error: {e}")

            st.session_state["tutor_quiz"] = quiz_data
            st.session_state["tutor_quiz_feedback"] = ""

        quiz_data = st.session_state.get("tutor_quiz")

        if quiz_data:
            st.markdown(f"**Pregunta:** {quiz_data.get('pregunta', '')}")
            opciones = quiz_data.get("opciones") or []
            if opciones:
                respuesta_usuario = st.radio(
                    "Elige tu respuesta:",
                    opciones,
                    key="tutor_quiz_user_option"
                )

                if st.button("✅ Comprobar respuesta"):
                    correcta = quiz_data.get("respuesta_correcta", "")
                    explicacion = quiz_data.get("explicacion", "")
                    if respuesta_usuario == correcta:
                        feedback = "✅ ¡Correcto! " + (explicacion or "")
                    else:
                        feedback = (
                            f"❌ No es correcto. La respuesta correcta era: **{correcta}**. "
                            + (explicacion or "")
                        )
                    st.session_state["tutor_quiz_feedback"] = feedback

        if st.session_state.get("tutor_quiz_feedback"):
            st.markdown(st.session_state["tutor_quiz_feedback"])

# =========================================================
# TAB 6: SENTIMIENTO & NOTICIAS (IA)
# =========================================================
with tab6:
    st.subheader("📰 Sentimiento & Noticias del portafolio (IA)")

    client = get_openai_client()
    if client is None:
        st.warning(
            "Configura tu API key dentro de la función get_openai_client() "
            "para usar el análisis de sentimiento."
        )
    else:
        st.markdown("#### Resumen numérico rápido")

        metrics_rows_news = []
        for t in tickers:
            df_t = dict_hist.get(t)
            if df_t is None or df_t.empty:
                continue
            info = fetch_static_info(t)
            sector = info.get("sector") or "Sin sector"
            country = info.get("country") or "—"
            series = df_t["Adj Close"]
            ann_ret, ann_vol, sharpe = annualized_metrics(series, rf_annual=rf)
            metrics_rows_news.append({
                "Ticker": t,
                "País": country,
                "Sector": sector,
                "Retorno": ann_ret,
                "Volatilidad": ann_vol,
                "Sharpe": sharpe
            })

        if metrics_rows_news:
            df_news_metrics = pd.DataFrame(metrics_rows_news)
            fmt_news = {
                "Retorno": lambda x: "—" if pd.isna(x) else f"{x*100:,.2f}%",
                "Volatilidad": lambda x: "—" if pd.isna(x) else f"{x*100:,.2f}%",
                "Sharpe": lambda x: "—" if pd.isna(x) else f"{x:,.3f}",
            }
            st.dataframe(df_news_metrics.style.format(fmt_news), use_container_width=True, height=260)
        else:
            st.info("No hay suficientes datos para mostrar métricas generales.")

        st.markdown("#### Noticias recientes por ticker (si Yahoo las ofrece)")

        all_news_text_parts = []
        for t in tickers:
            news_list = fetch_yf_news_simple(t, max_items=3)
            with st.expander(f"📰 {t} – noticias recientes"):
                if not news_list:
                    st.write("Sin noticias disponibles desde yfinance para este ticker.")
                else:
                    for item in news_list:
                        title = item["title"]
                        link = item["link"]
                        publisher = item["publisher"] or "Fuente desconocida"
                        if link:
                            st.markdown(f"- [{title}]({link}) — *{publisher}*")
                        else:
                            st.markdown(f"- {title} — *{publisher}*")
                        all_news_text_parts.append(f"{t}: {title} ({publisher})")

        base_context = build_ai_context_for_tickers(tickers, dict_hist, rf)
        stats_text = build_portfolio_stats_text(tickers, dict_hist, closes_df, rf)
        news_context = "\n".join(all_news_text_parts) if all_news_text_parts else "No se recuperaron noticias para estos tickers."

        st.markdown("---")
        st.markdown("#### Análisis de sentimiento con IA")

        if "sentiment_ai_text" not in st.session_state:
            st.session_state["sentiment_ai_text"] = ""

        if st.button("🧠 Analizar sentimiento del portafolio"):
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un analista de mercados. Recibirás datos de un portafolio "
                                "y algunas noticias recientes (si existen). Debes:\n"
                                "1) Describir el sentimiento general (positivo, neutral, mixto, negativo) "
                                "hacia el portafolio.\n"
                                "2) Mencionar los principales riesgos que ves.\n"
                                "3) Mencionar posibles oportunidades.\n"
                                "4) No des recomendaciones directas de compra/venta.\n"
                                "Responde en español, bien estructurado."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Este es el contexto numérico de los tickers:\n"
                                f"{base_context}\n\n"
                                "Resumen agregado del portafolio:\n"
                                f"{stats_text}\n\n"
                                "Estas son algunas noticias recientes asociadas a los tickers "
                                "(si se encontraron):\n"
                                f"{news_context}\n\n"
                                "Con todo esto, haz un análisis de sentimiento general del portafolio."
                            ),
                        },
                    ],
                )
                sentiment_text = completion.choices[0].message.content.strip()
            except Exception as e:
                sentiment_text = f"No se pudo generar el análisis de sentimiento. Error: {e}"

            st.session_state["sentiment_ai_text"] = sentiment_text

        if st.session_state.get("sentiment_ai_text"):
            st.markdown("#### Resultado del análisis de sentimiento")
            st.write(st.session_state["sentiment_ai_text"])

# =========================================================
# TAB 7: SCANNER DE OPORTUNIDADES (IA)
# =========================================================
with tab7:
    st.subheader("🔍 Scanner de oportunidades dentro de mis tickers (IA)")

    client = get_openai_client()
    if client is None:
        st.warning(
            "Configura tu API key dentro de la función get_openai_client() "
            "para usar el scanner de oportunidades."
        )
    else:
        metrics_rows_scan = []
        for t in tickers:
            df_t = dict_hist.get(t)
            if df_t is None or df_t.empty:
                continue
            info = fetch_static_info(t)
            sector = info.get("sector") or "Sin sector"
            series = df_t["Adj Close"]
            ann_ret, ann_vol, sharpe = annualized_metrics(series, rf_annual=rf)
            metrics_rows_scan.append({
                "Ticker": t,
                "Sector": sector,
                "Retorno": ann_ret,
                "Volatilidad": ann_vol,
                "Sharpe": sharpe
            })

        if not metrics_rows_scan:
            st.info("No hay suficientes datos para analizar oportunidades.")
        else:
            df_scan = pd.DataFrame(metrics_rows_scan)

            st.markdown("#### Métricas base de tus acciones")
            fmt_scan = {
                "Retorno": lambda x: "—" if pd.isna(x) else f"{x*100:,.2f}%",
                "Volatilidad": lambda x: "—" if pd.isna(x) else f"{x*100:,.2f}%",
                "Sharpe": lambda x: "—" if pd.isna(x) else f"{x:,.3f}",
            }
            st.dataframe(df_scan.style.format(fmt_scan), use_container_width=True, height=260)

            st.markdown("---")
            st.markdown("#### ¿Qué tipo de oportunidad estás buscando?")

            preferencia = st.text_area(
                "Describe el tipo de acción/oportunidad que te interesa:",
                value=(
                    "Quiero oportunidades de crecimiento en tecnología, con retorno esperado alto, "
                    "aunque acepto algo de volatilidad. Prefiero acciones con buen Sharpe si es posible."
                )
            )

            if "scanner_ai_text" not in st.session_state:
                st.session_state["scanner_ai_text"] = ""

            if st.button("🔎 Analizar oportunidades con IA"):
                if preferencia.strip():
                    try:
                        texto_scanner = scanner_ai_suggestions(
                            client,
                            df_metrics=df_scan,
                            preference_text=preferencia.strip()
                        )
                    except Exception as e:
                        texto_scanner = f"No se pudo generar el análisis de oportunidades. Error: {e}"

                    st.session_state["scanner_ai_text"] = texto_scanner

            if st.session_state.get("scanner_ai_text"):
                st.markdown("#### Análisis de oportunidades dentro de tus tickers")
                st.write(st.session_state["scanner_ai_text"])
