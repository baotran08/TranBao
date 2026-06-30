import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from pathlib import Path

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Tối ưu hóa Danh mục Đầu tư EMA + RSI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thử nghiệm import scipy
try:
    from scipy.stats import ttest_1samp, wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ==========================================================================================
# CẤU HÌNH GIAO DIỆN PREMIUM
# ==========================================================================================
st.markdown("""
<style>
    /* Premium aesthetics */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .gradient-text {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 8px 12px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #8b949e;
        font-weight: 600;
        border: none;
        padding: 0 16px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #30363d !important;
        color: #58a6ff !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================================================
# 0. HÀM ĐỌC VÀ XỬ LÝ DỮ LIỆU (CÓ CACHE)
# ==========================================================================================
@st.cache_data(show_spinner="Đang đọc dữ liệu lịch sử...")
def load_pivots(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    # Loại bỏ các cột Unnamed
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["date", "ticker"])
    df = df.sort_values(["ticker", "date"])

    # Ưu tiên giá điều chỉnh. Nếu không có adj_open/adj_close thì dùng open/close.
    close_col = "adj_close" if "adj_close" in df.columns else "close"
    open_col  = "adj_open"  if "adj_open"  in df.columns else "open"

    close = df.pivot(index="date", columns="ticker", values=close_col).sort_index()
    open_ = df.pivot(index="date", columns="ticker", values=open_col).sort_index()

    return close, open_

# ==========================================================================================
# 1. HÀM CHỈ BÁO KỸ THUẬT & TÍN HIỆU
# ==========================================================================================
def rsi_matrix(close_df, window):
    delta = close_df.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def ema_rsi_signal_matrix(close_df, p):
    ema_fast = close_df.ewm(span=p["ema_fast"], adjust=False).mean()
    ema_slow = close_df.ewm(span=p["ema_slow"], adjust=False).mean()
    rsi = rsi_matrix(close_df, p["rsi_window"])

    trend_up = ema_fast > ema_slow
    buy = trend_up & (rsi > p["rsi_lower"]) & (rsi < p["rsi_upper"])
    sell = (ema_fast < ema_slow) | (rsi > p["rsi_upper"])

    buy = buy & (~sell)
    return buy.fillna(False), sell.fillna(False)

def ema_rsi_signals_single(close_series, p):
    # Dành cho 1 cổ phiếu cụ thể
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/p["rsi_window"], adjust=False, min_periods=p["rsi_window"]).mean()
    avg_loss = loss.ewm(alpha=1/p["rsi_window"], adjust=False, min_periods=p["rsi_window"]).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)

    ema_fast = close_series.ewm(span=p["ema_fast"], adjust=False).mean()
    ema_slow = close_series.ewm(span=p["ema_slow"], adjust=False).mean()

    trend_up = ema_fast > ema_slow
    buy = trend_up & (rsi > p["rsi_lower"]) & (rsi < p["rsi_upper"])
    sell = (ema_fast < ema_slow) | (rsi > p["rsi_upper"])
    buy = buy & (~sell)

    pos = pd.Series(np.nan, index=close_series.index)
    pos = pos.mask(buy, 1.0).mask(sell, 0.0).ffill().fillna(0.0)
    return pos, buy, sell, ema_fast, ema_slow, rsi

# ==========================================================================================
# 2. HÀM BACKTEST
# ==========================================================================================
def backtest_single_ema_rsi(open_arr, close_arr, buy_arr, sell_arr,
                             stop_loss=0.10, capital=1_000_000_000, fee=0.0015):
    n = len(close_arr)
    equity = np.full(n, np.nan, dtype=float)
    cash = float(capital)
    shares = 0.0
    inpos = False
    entry_price = np.nan
    trades = 0
    fees_paid = 0.0

    equity[0] = capital
    for t in range(1, n):
        o_t, c_t, c_prev = open_arr[t], close_arr[t], close_arr[t-1]
        if not np.isfinite(o_t) or not np.isfinite(c_t):
            equity[t] = equity[t-1]
            continue

        prev_buy = bool(buy_arr[t-1])
        prev_sell = bool(sell_arr[t-1])
        stop_hit = inpos and np.isfinite(entry_price) and np.isfinite(c_prev) and (c_prev < entry_price * (1 - stop_loss))

        if inpos and (prev_sell or stop_hit):
            proceeds = shares * o_t
            f = proceeds * fee
            cash = proceeds - f
            shares = 0.0
            inpos = False
            entry_price = np.nan
            trades += 1
            fees_paid += f

        if (not inpos) and prev_buy and cash > 0:
            f = cash * fee
            shares = (cash - f) / o_t
            cash = 0.0
            inpos = True
            entry_price = o_t
            trades += 1
            fees_paid += f

        equity[t] = cash + shares * c_t

    return equity, trades, fees_paid

def backtest_matrix_ema_rsi(open_df, close_df, buy_df, sell_df, p,
                             capital=1_000_000_000, fee=0.0015):
    O = open_df.values.astype(float)
    C = close_df.values.astype(float)
    BUY = buy_df.values.astype(bool)
    SELL = sell_df.values.astype(bool)

    T, N = C.shape
    equity = np.full((T, N), np.nan, dtype=float)
    cash = np.full(N, float(capital))
    shares = np.zeros(N, dtype=float)
    inpos = np.zeros(N, dtype=bool)
    entry_price = np.full(N, np.nan, dtype=float)
    trades = np.zeros(N, dtype=int)
    fees_paid = np.zeros(N, dtype=float)

    equity[0, :] = capital
    for t in range(1, T):
        o_t = O[t, :]
        c_t = C[t, :]
        c_prev = C[t-1, :]
        valid = np.isfinite(o_t) & np.isfinite(c_t)

        stop_hit = inpos & np.isfinite(entry_price) & np.isfinite(c_prev) & (c_prev < entry_price * (1 - p["stop_loss"]))
        exit_mask = valid & inpos & (SELL[t-1, :] | stop_hit)

        if exit_mask.any():
            proceeds = shares[exit_mask] * o_t[exit_mask]
            f = proceeds * fee
            cash[exit_mask] = proceeds - f
            shares[exit_mask] = 0.0
            inpos[exit_mask] = False
            entry_price[exit_mask] = np.nan
            trades[exit_mask] += 1
            fees_paid[exit_mask] += f

        enter_mask = valid & (~inpos) & BUY[t-1, :] & (cash > 0)
        if enter_mask.any():
            f = cash[enter_mask] * fee
            shares[enter_mask] = (cash[enter_mask] - f) / o_t[enter_mask]
            cash[enter_mask] = 0.0
            inpos[enter_mask] = True
            entry_price[enter_mask] = o_t[enter_mask]
            trades[enter_mask] += 1
            fees_paid[enter_mask] += f

        marked_value = cash + shares * np.where(np.isfinite(c_t), c_t, np.nan)
        prev_value = equity[t-1, :]
        equity[t, :] = np.where(valid, marked_value, prev_value)

    return equity, trades, fees_paid

# ==========================================================================================
# 3. CHỈ TIÊU ĐÁNH GIÁ
# ==========================================================================================
def perf_metrics(equity, n_trades=None, initial=1_000_000_000,
                 rf_daily=0.04/252, periods=252):
    equity = np.asarray(equity, dtype=float)
    equity = equity[np.isfinite(equity)]
    if len(equity) < 2:
        out = {
            "Total Return [%]": np.nan, "CAGR [%]": np.nan, "Volatility [%]": np.nan,
            "Sharpe": np.nan, "Sortino": np.nan, "Max Drawdown [%]": np.nan,
            "Calmar": np.nan, "Final Value": np.nan,
        }
        if n_trades is not None:
            out["Trades"] = n_trades
        return out

    ret = equity[1:] / equity[:-1] - 1.0
    total_return = equity[-1] / initial - 1.0
    years = len(equity) / periods
    cagr = (equity[-1] / initial) ** (1 / years) - 1.0 if equity[-1] > 0 else -1.0
    vol = ret.std(ddof=1) * np.sqrt(periods) if len(ret) > 1 else np.nan
    excess = ret - rf_daily
    sd = ret.std(ddof=1)
    sharpe = np.sqrt(periods) * excess.mean() / sd if sd > 0 else np.nan
    downside = np.minimum(excess, 0.0)
    dd_dev = np.sqrt((downside ** 2).mean())
    sortino = np.sqrt(periods) * excess.mean() / dd_dev if dd_dev > 0 else np.nan
    run_max = np.maximum.accumulate(equity)
    mdd = (equity / run_max - 1.0).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    out = {
        "Total Return [%]": total_return * 100,
        "CAGR [%]": cagr * 100,
        "Volatility [%]": vol * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown [%]": mdd * 100,
        "Calmar": calmar,
        "Final Value": equity[-1],
    }
    if n_trades is not None:
        out["Trades"] = int(n_trades)
    return out

def sharpe_vector(equity, rf_daily=0.04/252, periods=252):
    ret = equity[1:] / equity[:-1] - 1.0
    excess = ret - rf_daily
    sd = np.nanstd(ret, axis=0, ddof=1)
    mean_excess = np.nanmean(excess, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(periods) * mean_excess / sd

# ==========================================================================================
# 4. THUẬT TOÁN PSO
# ==========================================================================================
PARAM_BOUNDS = [
    ("ema_fast",    5,   30,  True),
    ("ema_slow",   20,   90,  True),
    ("rsi_window",  7,   25,  True),
    ("rsi_lower",  20,   50,  False),
    ("rsi_upper",  60,   90,  False),
    ("stop_loss", 0.05, 0.20, False),
]

def decode(vec):
    p = {}
    for val, (name, lo, hi, is_int) in zip(vec, PARAM_BOUNDS):
        v = lo + (hi - lo) * val
        p[name] = int(round(v)) if is_int else float(v)

    if p["ema_slow"] <= p["ema_fast"]:
        p["ema_slow"] = p["ema_fast"] + 1
    if p["rsi_upper"] <= p["rsi_lower"]:
        p["rsi_upper"] = p["rsi_lower"] + 5
    p["stop_loss"] = float(np.clip(p["stop_loss"], 0.03, 0.30))
    return p

def pso_optimize(fitness_fn, n_particles=24, n_iter=35, seed=42,
                 w=0.7, c1=1.5, c2=1.5, progress_bar=None, status_text=None):
    rng = np.random.default_rng(seed)
    dim = len(PARAM_BOUNDS)
    X = rng.random((n_particles, dim))
    V = rng.uniform(-0.1, 0.1, (n_particles, dim))

    pbest = X.copy()
    pbest_val = np.array([fitness_fn(decode(x)) for x in X])
    g = int(np.nanargmax(pbest_val))
    gbest = pbest[g].copy()
    gbest_val = pbest_val[g]

    for it in range(n_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -0.3, 0.3)
        X = np.clip(X + V, 0.0, 1.0)

        vals = np.array([fitness_fn(decode(x)) for x in X])
        imp = vals > pbest_val
        pbest[imp] = X[imp]
        pbest_val[imp] = vals[imp]

        if np.nanmax(pbest_val) > gbest_val:
            g = int(np.nanargmax(pbest_val))
            gbest = pbest[g].copy()
            gbest_val = pbest_val[g]

        if progress_bar is not None:
            progress_bar.progress((it + 1) / n_iter)
        if status_text is not None:
            status_text.write(f"Vòng PSO {it + 1}/{n_iter} | Sharpe in-sample tốt nhất = {gbest_val:.4f}")

    return decode(gbest), float(gbest_val)

# ==========================================================================================
# 5. HÀM MÔ PHỎNG DANH MỤC & REBALANCE
# ==========================================================================================
def rebalance_indices(dates, freq, oos_start_pos):
    if freq == "none":
        return set()
    idx = []
    for pos in range(oos_start_pos + 1, len(dates)):
        d, prev = dates[pos], dates[pos - 1]
        if freq == "monthly" and (d.year, d.month) != (prev.year, prev.month):
            idx.append(pos)
        elif freq == "quarterly" and (d.year, (d.month - 1)//3) != (prev.year, (prev.month - 1)//3):
            idx.append(pos)
        elif freq == "annual" and d.year != prev.year:
            idx.append(pos)
    return set(idx)

def compute_weights(scheme, perf_vector, n):
    if scheme == "equal":
        return np.full(n, 1.0 / n)
    pos = np.clip(np.asarray(perf_vector, dtype=float), 0.0, None)
    return np.full(n, 1.0 / n) if pos.sum() <= 0 else pos / pos.sum()

def run_portfolio_ema_rsi(close_df, open_df, stocks, params, weight_scheme="equal", rebalance="quarterly",
                           init_perf=None, capital=1_000_000_000, fee=0.0015, oos_start="2021-01-04"):
    dates = close_df.index
    oos_ts = pd.Timestamp(oos_start)
    if oos_ts not in dates:
        # Lấy ngày đầu tiên sau mốc thời gian OOS bắt đầu
        post_dates = dates[dates >= oos_ts]
        if len(post_dates) == 0:
            st.error("Mốc Out-of-sample start nằm ngoài phạm vi dữ liệu!")
            return None, None
        oos_ts = post_dates[0]
    oos_pos = list(dates).index(oos_ts)

    N = len(stocks)
    buy, sell = ema_rsi_signal_matrix(close_df[stocks], params)
    BUY = buy.values.astype(bool)
    SELL = sell.values.astype(bool)
    C = close_df[stocks].values.astype(float)
    O = open_df[stocks].values.astype(float)
    rebal = rebalance_indices(dates, rebalance, oos_pos)

    if init_perf is None:
        init_perf = {tk: 1.0 for tk in stocks}
    init_perf_vec = np.array([init_perf.get(tk, 1.0) for tk in stocks], dtype=float)
    w0 = compute_weights(weight_scheme, init_perf_vec, N)

    cash = w0 * capital
    shares = np.zeros(N, dtype=float)
    inpos = np.zeros(N, dtype=bool)
    entry_price = np.full(N, np.nan, dtype=float)
    last_rebal_value = w0 * capital

    eq_dates, eq_vals = [], []
    total_trades = 0
    total_fee_paid = 0.0
    rebal_count = 0

    for pos in range(oos_pos, len(dates)):
        o_t = O[pos, :]
        c_t = C[pos, :]
        c_prev = C[pos - 1, :] if pos > 0 else np.full(N, np.nan)
        valid = np.isfinite(o_t) & np.isfinite(c_t)

        prev_buy = BUY[pos - 1, :] if pos > 0 else np.zeros(N, dtype=bool)
        prev_sell = SELL[pos - 1, :] if pos > 0 else np.zeros(N, dtype=bool)
        stop_hit = inpos & np.isfinite(entry_price) & np.isfinite(c_prev) & (c_prev < entry_price * (1 - params["stop_loss"]))

        exit_mask = valid & inpos & (prev_sell | stop_hit)
        if exit_mask.any():
            proceeds = shares[exit_mask] * o_t[exit_mask]
            f = proceeds * fee
            cash[exit_mask] = proceeds - f
            shares[exit_mask] = 0.0
            inpos[exit_mask] = False
            entry_price[exit_mask] = np.nan
            total_trades += int(exit_mask.sum())
            total_fee_paid += float(f.sum())

        enter_mask = valid & (~inpos) & prev_buy & (cash > 0)
        if enter_mask.any():
            f = cash[enter_mask] * fee
            shares[enter_mask] = (cash[enter_mask] - f) / o_t[enter_mask]
            cash[enter_mask] = 0.0
            inpos[enter_mask] = True
            entry_price[enter_mask] = o_t[enter_mask]
            total_trades += int(enter_mask.sum())
            total_fee_paid += float(f.sum())

        if pos in rebal:
            sleeve_val = cash + shares * c_t
            V = float(np.nansum(sleeve_val))
            if V > 0:
                if weight_scheme == "performance":
                    growth = sleeve_val / np.where(last_rebal_value > 0, last_rebal_value, np.nan) - 1.0
                    growth = np.nan_to_num(growth, nan=0.0)
                    w = compute_weights("performance", growth, N)
                else:
                    w = compute_weights("equal", None, N)

                target = w * V
                reb_fee = 0.0
                for i in range(N):
                    if inpos[i] and np.isfinite(c_t[i]):
                        current_stock_value = shares[i] * c_t[i]
                        reb_fee += fee * abs(target[i] - current_stock_value)
                scale = (V - reb_fee) / V if V > 0 else 1.0

                for i in range(N):
                    if inpos[i] and np.isfinite(c_t[i]):
                        shares[i] = (target[i] * scale) / c_t[i]
                        cash[i] = 0.0
                    else:
                        shares[i] = 0.0
                        cash[i] = target[i] * scale
                        inpos[i] = False
                        entry_price[i] = np.nan

                total_fee_paid += float(reb_fee)
                last_rebal_value = target * scale
                rebal_count += 1

        total_value = float(np.nansum(cash + shares * c_t))
        eq_dates.append(dates[pos])
        eq_vals.append(total_value)

    equity = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates), name="EMA_RSI_Strategy")
    m = perf_metrics(equity.values, total_trades, initial=capital)
    m["Fees Paid"] = total_fee_paid
    m["Rebalance Count"] = rebal_count
    return equity, m

# ==========================================================================================
# 6. BENCHMARK
# ==========================================================================================
def buy_hold_basket(close_df, open_df, stocks, weights=None, capital=1_000_000_000,
                     fee=0.0015, oos_start="2021-01-04", name="BuyHold"):
    dates = close_df.index
    oos_ts = pd.Timestamp(oos_start)
    if oos_ts not in dates:
        oos_ts = dates[dates >= oos_ts][0]
    oos_pos = list(dates).index(oos_ts)

    N = len(stocks)
    if weights is None:
        weights = np.full(N, 1.0 / N)
    o0 = open_df[stocks].values[oos_pos]
    shares = (weights * capital * (1 - fee)) / o0
    C = close_df[stocks].values[oos_pos:]
    equity = pd.Series((C * shares).sum(axis=1), index=dates[oos_pos:], name=name)
    return equity, perf_metrics(equity.values, 1, initial=capital)

def buy_hold_index(close_df, ticker="VNINDEX", capital=1_000_000_000, oos_start="2021-01-04"):
    dates = close_df.index
    oos_ts = pd.Timestamp(oos_start)
    if oos_ts not in dates:
        oos_ts = dates[dates >= oos_ts][0]
    oos_pos = list(dates).index(oos_ts)
    s = close_df[ticker].values[oos_pos:]
    equity = pd.Series(capital * s / s[0], index=dates[oos_pos:], name=ticker)
    return equity, perf_metrics(equity.values, 0, initial=capital)

def submetrics(equity, label, initial_capital=1_000_000_000):
    out = {"label": label, "full": perf_metrics(equity.values, initial=initial_capital)}
    for yr in [2021, 2022, 2023]:
        seg = equity[equity.index.year == yr]
        prev = equity[equity.index.year < yr]
        start_val = prev.values[-1] if len(prev) else initial_capital
        e = np.concatenate([[start_val], seg.values]) if len(seg) else np.array([start_val])
        out[str(yr)] = perf_metrics(e, initial=start_val)
    return out

# ==========================================================================================
# GIAO DIỆN CHÍNH STREAMLIT
# ==========================================================================================
st.markdown('<div class="gradient-text">Tối ưu hóa Danh mục Đầu tư EMA + RSI</div>', unsafe_allow_html=True)
st.write("Ứng dụng tối ưu hóa và backtest danh mục cổ phiếu Việt Nam sử dụng chỉ báo EMA & RSI kết hợp thuật toán tối ưu hóa bầy đàn (PSO).")

# --- SIDEBAR: Cấu hình chung ---
st.sidebar.header("⚙️ Cấu hình chung")

# Upload/Chọn nguồn dữ liệu
data_source = st.sidebar.radio("Nguồn dữ liệu:", ["Sử dụng file HOSE_2020_2023.csv mặc định", "Tải lên file CSV mới"])
csv_path = "HOSE_2020_2023.csv"

if data_source == "Tải lên file CSV mới":
    uploaded_file = st.sidebar.file_uploader("Chọn file dữ liệu CSV", type=["csv"])
    if uploaded_file is not None:
        csv_path = uploaded_file
        # Lưu file tạm nếu cần thiết hoặc đọc trực tiếp
    else:
        st.sidebar.warning("Đang sử dụng file mặc định vì chưa tải file mới.")

# Load dữ liệu
try:
    close, open_ = load_pivots(csv_path)
    STOCKS = [c for c in close.columns if c != "VNINDEX"]
    HAS_DATA = True
except Exception as e:
    st.error(f"Lỗi khi đọc file dữ liệu: {e}")
    HAS_DATA = False

if HAS_DATA:
    st.sidebar.subheader("💵 Thông số danh mục")
    initial_capital = st.sidebar.number_input("Vốn ban đầu (VND):", value=1_000_000_000, step=100_000_000)
    fee_rate = st.sidebar.slider("Phí giao dịch (%):", min_value=0.0, max_value=1.0, value=0.15, step=0.05) / 100
    risk_free_annual = st.sidebar.slider("Lãi suất phi rủi ro năm (%):", min_value=0.0, max_value=15.0, value=4.0, step=0.5) / 100
    trading_days = st.sidebar.number_input("Số ngày giao dịch/năm:", value=252)
    rf_daily = risk_free_annual / trading_days

    st.sidebar.subheader("📅 Phân chia thời gian")
    min_date = close.index.min().to_pydatetime()
    max_date = close.index.max().to_pydatetime()
    
    # In-Sample
    is_end_date = st.sidebar.date_input("Ngày kết thúc In-Sample (Học):", value=pd.Timestamp("2020-12-31"), min_value=min_date, max_value=max_date)
    # Out-of-Sample
    oos_start_date = st.sidebar.date_input("Ngày bắt đầu Out-of-Sample (Kiểm thử):", value=pd.Timestamp("2021-01-04"), min_value=min_date, max_value=max_date)

    st.sidebar.subheader("🏆 Chọn lọc")
    num_top_stocks = st.sidebar.number_input("Số cổ phiếu tuyển chọn (Top N):", min_value=1, max_value=len(STOCKS), value=3)

    # Khởi tạo tham số mặc định trong st.session_state
    if "best_p" not in st.session_state:
        st.session_state.best_p = {
            "ema_fast": 12,
            "ema_slow": 26,
            "rsi_window": 14,
            "rsi_lower": 30.0,
            "rsi_upper": 70.0,
            "stop_loss": 0.10
        }
    if "is_optimized" not in st.session_state:
        st.session_state.is_optimized = False

    # Định nghĩa các tab
    tab_overview, tab_pso, tab_rank, tab_oos, tab_stock_detail, tab_stats = st.tabs([
        "📊 Tổng quan Dữ liệu",
        "🧬 Tối ưu hóa PSO",
        "🏅 Lọc cổ phiếu (In-Sample)",
        "📈 Backtest Danh mục (OOS)",
        "🔍 Chi tiết từng cổ phiếu",
        "🧪 Kiểm định Thống kê"
    ])

    # ==========================================================================================
    # TAB 1: TỔNG QUAN DỮ LIỆU
    # ==========================================================================================
    with tab_overview:
        st.subheader("Thông tin bộ dữ liệu lịch sử")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tổng số ngày giao dịch", len(close.index))
        with col2:
            st.metric("Tổng số mã cổ phiếu", len(close.columns))
        with col3:
            st.metric("Có VN-Index trong dữ liệu", "Có" if "VNINDEX" in close.columns else "Không")
            
        st.write(f"**Khoảng thời gian:** {close.index.min().date()}  👉  {close.index.max().date()}")

        # Preview dữ liệu
        st.subheader("Bảng dữ liệu giá đóng cửa điều chỉnh (Preview)")
        st.dataframe(close.head(10), use_container_width=True)

        # Vẽ giá cổ phiếu tự chọn
        st.subheader("Biểu đồ giá lịch sử của các mã")
        selected_plot_stocks = st.multiselect("Chọn mã cổ phiếu cần vẽ biểu đồ giá:", options=list(close.columns), default=["VNINDEX"] if "VNINDEX" in close.columns else [STOCKS[0]])
        
        if selected_plot_stocks:
            fig = go.Figure()
            for tk in selected_plot_stocks:
                fig.add_trace(go.Scatter(x=close.index, y=close[tk], mode='lines', name=tk))
            fig.update_layout(
                title="Lịch sử giá đóng cửa (Chuẩn hóa/Gốc)",
                xaxis_title="Thời gian",
                yaxis_title="Giá (VND)",
                template="plotly_dark",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================================
    # TAB 2: TỐI ƯU HÓA PSO
    # ==========================================================================================
    with tab_pso:
        st.subheader("Tối ưu hóa Tham số Chiến lược bằng Thuật toán Bầy đàn (PSO)")
        st.markdown("""
        Thuật toán **PSO** sẽ tìm kiếm bộ tham số tối ưu trên giai đoạn dữ liệu **In-Sample** (Học) để tối đa hóa chỉ số Sharpe trung bình của các cổ phiếu:
        - **ema_fast**: Chu kỳ EMA nhanh (5 - 30 ngày)
        - **ema_slow**: Chu kỳ EMA chậm (20 - 90 ngày)
        - **rsi_window**: Chu kỳ tính RSI (7 - 25 ngày)
        - **rsi_lower**: Ngưỡng RSI mua vào (20 - 50)
        - **rsi_upper**: Ngưỡng RSI bán ra (60 - 90)
        - **stop_loss**: Ngưỡng cắt lỗ (5% - 20%)
        """)

        # Phân mảnh dữ liệu In-sample
        is_mask = close.index <= pd.Timestamp(is_end_date)
        close_is = close.loc[is_mask, STOCKS]
        open_is = open_.loc[is_mask, STOCKS]

        st.info(f"Thời gian học In-Sample: **{close_is.index.min().date()}** đến **{close_is.index.max().date()}** ({len(close_is)} phiên).")

        col_pso1, col_pso2 = st.columns(2)
        with col_pso1:
            st.write("📊 **Cấu hình tìm kiếm PSO**")
            pso_particles = st.slider("Số lượng phần tử bầy đàn (particles):", 10, 50, 24)
            pso_iter = st.slider("Số lượng vòng lặp tối ưu (iterations):", 10, 100, 35)
            pso_seed = st.number_input("Seed ngẫu nhiên:", value=42)

            run_optimization = st.button("🚀 Chạy tối ưu hóa PSO")

            if run_optimization:
                def fitness(p):
                    buy, sell = ema_rsi_signal_matrix(close_is, p)
                    eq, _, _ = backtest_matrix_ema_rsi(open_is, close_is, buy, sell, p, capital=initial_capital, fee=fee_rate)
                    sh = sharpe_vector(eq, rf_daily=rf_daily, periods=trading_days)
                    sh = sh[np.isfinite(sh)]
                    return float(np.nanmean(sh)) if len(sh) else -9.0

                progress_bar = st.progress(0)
                status_text = st.empty()

                with st.spinner("Đang chạy thuật toán tối ưu hóa PSO..."):
                    best_p, best_val = pso_optimize(
                        fitness,
                        n_particles=pso_particles,
                        n_iter=pso_iter,
                        seed=pso_seed,
                        progress_bar=progress_bar,
                        status_text=status_text
                    )
                
                st.session_state.best_p = best_p
                st.session_state.is_optimized = True
                st.session_state.best_sharpe = best_val
                st.success("Tối ưu hóa PSO hoàn tất!")

        with col_pso2:
            st.write("🔧 **Bộ tham số hiện tại (Dùng cho Backtest)**")
            
            p_fast = st.number_input("ema_fast (EMA nhanh):", min_value=3, max_value=50, value=int(st.session_state.best_p["ema_fast"]))
            p_slow = st.number_input("ema_slow (EMA chậm):", min_value=15, max_value=120, value=int(st.session_state.best_p["ema_slow"]))
            p_rsi_w = st.number_input("rsi_window (Chu kỳ RSI):", min_value=5, max_value=50, value=int(st.session_state.best_p["rsi_window"]))
            p_rsi_l = st.number_input("rsi_lower (Ngưỡng RSI dưới):", min_value=10.0, max_value=60.0, value=float(st.session_state.best_p["rsi_lower"]))
            p_rsi_u = st.number_input("rsi_upper (Ngưỡng RSI trên):", min_value=50.0, max_value=95.0, value=float(st.session_state.best_p["rsi_upper"]))
            p_sl = st.slider("stop_loss (Cắt lỗ):", min_value=0.01, max_value=0.50, value=float(st.session_state.best_p["stop_loss"]), step=0.01)

            # Cập nhật lại cấu hình nếu thay đổi thủ công
            st.session_state.best_p = {
                "ema_fast": p_fast,
                "ema_slow": p_slow,
                "rsi_window": p_rsi_w,
                "rsi_lower": p_rsi_l,
                "rsi_upper": p_rsi_u,
                "stop_loss": p_sl
            }

            st.write("**Tham số chiến lược được áp dụng:**")
            st.code(json.dumps(st.session_state.best_p, indent=2))
            
            if st.session_state.is_optimized:
                st.write(f"🏆 *Sharpe trung bình in-sample đạt được trong tối ưu hóa: **{st.session_state.best_sharpe:.4f}***")

    # ==========================================================================================
    # TAB 3: CHỌN LỌC CỔ PHIẾU (IN-SAMPLE)
    # ==========================================================================================
    with tab_rank:
        st.subheader("Xếp hạng cổ phiếu trong giai đoạn học In-Sample")
        st.write("Sử dụng bộ tham số hiện tại để đánh giá hiệu quả của từng cổ phiếu trong năm 2020 (hoặc mốc In-Sample được cấu hình), xếp hạng để chọn ra các mã có Sharpe tốt nhất.")

        # Phân mảnh dữ liệu In-sample
        is_mask = close.index <= pd.Timestamp(is_end_date)
        close_is = close.loc[is_mask, STOCKS]
        open_is = open_.loc[is_mask, STOCKS]

        buy_is, sell_is = ema_rsi_signal_matrix(close_is, st.session_state.best_p)
        eqM, ntrM, feesM = backtest_matrix_ema_rsi(open_is, close_is, buy_is, sell_is, st.session_state.best_p, capital=initial_capital, fee=fee_rate)

        rows = []
        for j, tk in enumerate(STOCKS):
            m = perf_metrics(eqM[:, j], int(ntrM[j]), initial=initial_capital, rf_daily=rf_daily, periods=trading_days)
            rows.append({
                "Ticker": tk,
                "Lợi nhuận In-sample (%)": m["Total Return [%]"],
                "Sharpe In-sample": m["Sharpe"],
                "Sortino In-sample": m["Sortino"],
                "Max Drawdown (%)": m["Max Drawdown [%]"],
                "Số giao dịch": int(ntrM[j]),
                "Tổng phí giao dịch": float(feesM[j]),
            })

        rank = pd.DataFrame(rows).sort_values(["Sharpe In-sample", "Lợi nhuận In-sample (%)"], ascending=False).reset_index(drop=True)
        top_stocks_list = rank.head(int(num_top_stocks))["Ticker"].tolist()
        
        # Save top stocks for OOS tab
        st.session_state.top_stocks = top_stocks_list

        # Biểu diễn Top 10 mã
        st.write(f"Top 10 cổ phiếu xuất sắc nhất trên dữ liệu In-Sample:")
        st.dataframe(rank.head(10).style.format({
            "Lợi nhuận In-sample (%)": "{:.2f}%",
            "Sharpe In-sample": "{:.4f}",
            "Sortino In-sample": "{:.4f}",
            "Max Drawdown (%)": "{:.2f}%",
            "Số giao dịch": "{:d}",
            "Tổng phí giao dịch": "{:,.0f} VND"
        }), use_container_width=True)

        st.subheader("Danh mục cổ phiếu được tuyển chọn (Top N)")
        st.success(f"Dựa trên cấu hình chọn lọc Top **{num_top_stocks}**, các cổ phiếu được chọn vào danh mục là: **{top_stocks_list}**")

        # Vẽ đồ thị Sharpe của các mã
        fig_rank = px.bar(
            rank.head(20),
            x="Ticker",
            y="Sharpe In-sample",
            title="Sharpe Ratio của 20 mã hàng đầu (In-Sample)",
            color="Sharpe In-sample",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    # ==========================================================================================
    # TAB 4: BACKTEST DANH MỤC (OUT-OF-SAMPLE)
    # ==========================================================================================
    with tab_oos:
        st.subheader("Kiểm thử Danh mục đầu tư Out-of-Sample (OOS)")
        st.write("Mô phỏng hiệu quả giao dịch của danh mục tuyển chọn trong giai đoạn kiểm thử lịch sử.")

        if "top_stocks" not in st.session_state or not st.session_state.top_stocks:
            st.warning("Vui lòng chạy Tab 'Lọc cổ phiếu (In-Sample)' trước để chọn danh sách cổ phiếu tối ưu.")
        else:
            top_stocks = st.session_state.top_stocks
            st.info(f"Danh mục tuyển chọn (cố định từ In-sample): **{top_stocks}**")
            
            # Cấu hình danh mục OOS
            col_oos1, col_oos2 = st.columns(2)
            with col_oos1:
                weight_scheme = st.selectbox("Phương pháp chia tỷ trọng vốn:", ["equal", "performance"], format_func=lambda x: "Trọng số đều (Equal Weight)" if x == "equal" else "Theo hiệu quả In-Sample (Performance Weight)")
            with col_oos2:
                rebalance_freq = st.selectbox("Tần suất tái cân bằng danh mục:", ["none", "monthly", "quarterly", "annual"], format_func=lambda x: {
                    "none": "Không tái cân bằng (Buy & Hold vị thế)",
                    "monthly": "Tái cân bằng hàng tháng (Monthly)",
                    "quarterly": "Tái cân bằng hàng quý (Quarterly)",
                    "annual": "Tái cân bằng hàng năm (Annual)"
                }[x])

            # Tính toán hiệu quả In-Sample làm trọng số performance
            is_mask = close.index <= pd.Timestamp(is_end_date)
            close_is_top = close.loc[is_mask, top_stocks]
            open_is_top = open_.loc[is_mask, top_stocks]
            buy_is_top, sell_is_top = ema_rsi_signal_matrix(close_is_top, st.session_state.best_p)
            eqM_is, ntrM_is, _ = backtest_matrix_ema_rsi(open_is_top, close_is_top, buy_is_top, sell_is_top, st.session_state.best_p, capital=initial_capital, fee=fee_rate)
            
            is_ret = {}
            for idx, tk in enumerate(top_stocks):
                m_is = perf_metrics(eqM_is[:, idx], int(ntrM_is[idx]), initial=initial_capital, rf_daily=rf_daily, periods=trading_days)
                is_ret[tk] = m_is["Total Return [%]"]

            # Chạy mô phỏng chiến lược OOS
            eq_strat, m_strat = run_portfolio_ema_rsi(
                close, open_, top_stocks, st.session_state.best_p,
                weight_scheme=weight_scheme,
                rebalance=rebalance_freq,
                init_perf=is_ret,
                capital=initial_capital,
                fee=fee_rate,
                oos_start=str(oos_start_date)
            )

            if eq_strat is not None:
                # Chạy Benchmarks
                eq_bh_top3, m_bh_top3 = buy_hold_basket(close, open_, top_stocks, capital=initial_capital, fee=fee_rate, oos_start=str(oos_start_date), name="BuyHold_TopN")
                eq_1n_all, m_1n_all = buy_hold_basket(close, open_, STOCKS, capital=initial_capital, fee=fee_rate, oos_start=str(oos_start_date), name="EqualWeight_All")
                
                if "VNINDEX" in close.columns:
                    eq_vnindex, m_vnindex = buy_hold_index(close, "VNINDEX", capital=initial_capital, oos_start=str(oos_start_date))
                else:
                    eq_vnindex, m_vnindex = None, None

                # Hiển thị Metrics chính dưới dạng Card
                st.subheader("Chỉ số hiệu quả danh mục")
                
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Tổng lợi nhuận</div>
                        <div class="metric-value">{m_strat['Total Return [%]']:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">CAGR</div>
                        <div class="metric-value">{m_strat['CAGR [%]']:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{m_strat['Sharpe']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value">{m_strat['Max Drawdown [%]']:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Số lần Rebalance</div>
                        <div class="metric-value">{m_strat['Rebalance Count']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Vẽ biểu đồ so sánh giá trị danh mục
                st.subheader("Biểu đồ tăng trưởng tài sản (OOS)")
                fig_port = go.Figure()
                fig_port.add_trace(go.Scatter(x=eq_strat.index, y=eq_strat.values, name="Chiến lược EMA+RSI Danh mục", line=dict(color="#00f2fe", width=3.5)))
                fig_port.add_trace(go.Scatter(x=eq_bh_top3.index, y=eq_bh_top3.values, name=f"Buy & Hold Top {num_top_stocks}", line=dict(color="#ff9f43", dash='dash')))
                fig_port.add_trace(go.Scatter(x=eq_1n_all.index, y=eq_1n_all.values, name="1/N Toàn bộ cổ phiếu", line=dict(color="#28c76f", dash='dot')))
                if eq_vnindex is not None:
                    fig_port.add_trace(go.Scatter(x=eq_vnindex.index, y=eq_vnindex.values, name="VN-Index Benchmark", line=dict(color="#ea5455", dash='longdash')))
                
                fig_port.update_layout(
                    title="So sánh giá trị tài sản ròng qua các phiên giao dịch (OOS)",
                    xaxis_title="Thời gian",
                    yaxis_title="Giá trị tài sản (VND)",
                    template="plotly_dark",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_port, use_container_width=True)

                # Bảng so sánh các phương pháp
                st.subheader("Bảng so sánh chi tiết các chỉ số hiệu quả")
                items = [
                    ("Chiến lược EMA+RSI (Danh mục)", m_strat),
                    (f"Buy & Hold Top {num_top_stocks} cổ phiếu", m_bh_top3),
                    ("1/N Toàn bộ cổ phiếu", m_1n_all),
                ]
                if m_vnindex is not None:
                    items.append(("VN-Index Benchmark", m_vnindex))
                
                sum_df = metrics_table = []
                for name, metrics in items:
                    sum_df.append({
                        "Phương pháp": name,
                        "Lợi nhuận tổng (%)": metrics["Total Return [%]"],
                        "CAGR (%)": metrics["CAGR [%]"],
                        "Độ biến động (%)": metrics["Volatility [%]"],
                        "Sharpe Ratio": metrics["Sharpe"],
                        "Sortino Ratio": metrics["Sortino"],
                        "Max Drawdown (%)": metrics["Max Drawdown [%]"],
                        "Calmar Ratio": metrics["Calmar"],
                        "Giá trị cuối cùng (VND)": metrics["Final Value"],
                        "Số giao dịch": metrics.get("Trades", 0),
                        "Phí giao dịch đã trả (VND)": metrics.get("Fees Paid", 0) if "Fees Paid" in metrics else (metrics.get("Fees paid", 0) if "Fees paid" in metrics else 0)
                    })
                
                summary_df = pd.DataFrame(sum_df)
                st.dataframe(summary_df.style.format({
                    "Lợi nhuận tổng (%)": "{:.2f}%",
                    "CAGR (%)": "{:.2f}%",
                    "Độ biến động (%)": "{:.2f}%",
                    "Sharpe Ratio": "{:.3f}",
                    "Sortino Ratio": "{:.3f}",
                    "Max Drawdown (%)": "{:.2f}%",
                    "Calmar Ratio": "{:.3f}",
                    "Giá trị cuối cùng (VND)": "{:,.0f} VND",
                    "Số giao dịch": "{:d}",
                    "Phí giao dịch đã trả (VND)": "{:,.0f} VND"
                }), use_container_width=True)

                # Phân tích theo năm
                st.subheader("Phân tích chi tiết theo năm")
                S_y = submetrics(eq_strat, "EMA+RSI", initial_capital=initial_capital)
                B_y = submetrics(eq_bh_top3, "Buy&Hold Top3", initial_capital=initial_capital)
                E_y = submetrics(eq_1n_all, "1/N Toàn bộ", initial_capital=initial_capital)
                V_y = submetrics(eq_vnindex, "VN-Index", initial_capital=initial_capital) if eq_vnindex is not None else None

                yearly_rows = []
                for yr in ["2021", "2022", "2023"]:
                    yearly_rows.append({
                        "Năm": yr,
                        "Chiến lược Lợi nhuận (%)": S_y[yr]["Total Return [%]"],
                        "Chiến lược MaxDD (%)": S_y[yr]["Max Drawdown [%]"],
                        "Buy&Hold Lợi nhuận (%)": B_y[yr]["Total Return [%]"],
                        "Buy&Hold MaxDD (%)": B_y[yr]["Max Drawdown [%]"],
                        "1/N All Lợi nhuận (%)": E_y[yr]["Total Return [%]"],
                        "VN-Index Lợi nhuận (%)": V_y[yr]["Total Return [%]"] if V_y is not None else np.nan,
                    })
                yearly_df = pd.DataFrame(yearly_rows)
                st.dataframe(yearly_df.style.format({
                    "Chiến lược Lợi nhuận (%)": "{:.2f}%",
                    "Chiến lược MaxDD (%)": "{:.2f}%",
                    "Buy&Hold Lợi nhuận (%)": "{:.2f}%",
                    "Buy&Hold MaxDD (%)": "{:.2f}%",
                    "1/N All Lợi nhuận (%)": "{:.2f}%",
                    "VN-Index Lợi nhuận (%)": "{:.2f}%"
                }), use_container_width=True)

                # Ma trận so sánh nhiều cấu hình
                st.subheader("Bảng so sánh chéo nhiều cấu hình quản lý vốn")
                st.write("Chạy so sánh hiệu suất với tất cả cấu hình phân bổ tỷ trọng (Trọng số đều vs Performance) và chu kỳ tái cân bằng (Không, Hàng tháng, Hàng quý, Hàng năm).")
                
                with st.spinner("Đang chạy kiểm thử lưới các cấu hình..."):
                    grid = []
                    for scheme in ["equal", "performance"]:
                        for reb in ["none", "monthly", "quarterly", "annual"]:
                            eq_tmp, m_tmp = run_portfolio_ema_rsi(
                                close, open_, top_stocks, st.session_state.best_p,
                                weight_scheme=scheme,
                                rebalance=reb,
                                init_perf=is_ret,
                                capital=initial_capital,
                                fee=fee_rate,
                                oos_start=str(oos_start_date)
                            )
                            if eq_tmp is not None:
                                grid.append({
                                    "Phân bổ tỷ trọng": "Trọng số đều (Equal)" if scheme == "equal" else "Hiệu quả In-sample (Performance)",
                                    "Chu kỳ tái cân bằng": {"none": "Không", "monthly": "Tháng", "quarterly": "Quý", "annual": "Năm"}[reb],
                                    "Lợi nhuận tổng (%)": m_tmp["Total Return [%]"],
                                    "CAGR (%)": m_tmp["CAGR [%]"],
                                    "Sharpe Ratio": m_tmp["Sharpe"],
                                    "Max Drawdown (%)": m_tmp["Max Drawdown [%]"],
                                    "Tổng giao dịch": m_tmp.get("Trades", 0),
                                    "Tổng phí giao dịch (VND)": m_tmp.get("Fees Paid", 0)
                                })
                    
                    grid_table = pd.DataFrame(grid).sort_values(["Sharpe Ratio", "Lợi nhuận tổng (%)"], ascending=False).reset_index(drop=True)
                    st.dataframe(grid_table.style.format({
                        "Lợi nhuận tổng (%)": "{:.2f}%",
                        "CAGR (%)": "{:.2f}%",
                        "Sharpe Ratio": "{:.3f}",
                        "Max Drawdown (%)": "{:.2f}%",
                        "Tổng giao dịch": "{:d}",
                        "Tổng phí giao dịch (VND)": "{:,.0f} VND"
                    }), use_container_width=True)

    # ==========================================================================================
    # TAB 5: PHÂN TÍCH TỪNG CỔ PHIẾU CỤ THỂ
    # ==========================================================================================
    with tab_stock_detail:
        st.subheader("Chi tiết Tín hiệu Giao dịch cho từng cổ phiếu")
        st.write("Xem lịch sử giao dịch, các điểm vào lệnh (Buy), thoát lệnh (Sell), và các thời điểm kích hoạt Cắt lỗ (Stop Loss) trên biểu đồ thực tế của từng mã.")

        active_stocks = st.session_state.top_stocks if ("top_stocks" in st.session_state and st.session_state.top_stocks) else STOCKS[:5]
        selected_stock = st.selectbox("Chọn cổ phiếu phân tích:", active_stocks)

        # Trích lọc dữ liệu Out-of-sample cho cổ phiếu được chọn
        oos_close = close.loc[close.index >= pd.Timestamp(oos_start_date), selected_stock].dropna()
        oos_open = open_.loc[close.index >= pd.Timestamp(oos_start_date), selected_stock].dropna()

        pos, buy_s, sell_s, ema_fast_s, ema_slow_s, rsi_s = ema_rsi_signals_single(oos_close, st.session_state.best_p)

        # Chạy backtest đơn lẻ để thu thập trạng thái Stop loss cụ thể
        equity_single, trades_single, fees_single = backtest_single_ema_rsi(
            oos_open.values, oos_close.values, buy_s.values, sell_s.values,
            stop_loss=st.session_state.best_p["stop_loss"],
            capital=initial_capital,
            fee=fee_rate
        )

        # Xác định chính xác vị trí bị Stop loss trong thực tế
        # (Tìm những điểm có vị thế hôm trước = 1, hôm nay = 0 nhưng không có tín hiệu bán từ RSI/EMA)
        in_pos_state = False
        entry_price = np.nan
        stop_hits = pd.Series(False, index=oos_close.index)
        for t in range(1, len(oos_close)):
            c_prev = oos_close.iloc[t-1]
            o_t = oos_open.iloc[t]
            prev_buy = buy_s.iloc[t-1]
            prev_sell = sell_s.iloc[t-1]

            if in_pos_state and (prev_sell or (np.isfinite(entry_price) and c_prev < entry_price * (1 - st.session_state.best_p["stop_loss"]))):
                # Bán
                if not prev_sell: # Bị stop loss
                    stop_hits.iloc[t] = True
                in_pos_state = False
                entry_price = np.nan

            if (not in_pos_state) and prev_buy:
                in_pos_state = True
                entry_price = o_t

        # Vẽ biểu đồ giá kèm tín hiệu
        fig_stock = go.Figure()
        fig_stock.add_trace(go.Scatter(x=oos_close.index, y=oos_close.values, name="Giá đóng cửa", line=dict(color="#ffffff", width=2)))
        fig_stock.add_trace(go.Scatter(x=ema_fast_s.index, y=ema_fast_s.values, name=f"EMA Nhanh ({st.session_state.best_p['ema_fast']})", line=dict(color="#00f2fe", width=1.5)))
        fig_stock.add_trace(go.Scatter(x=ema_slow_s.index, y=ema_slow_s.values, name=f"EMA Chậm ({st.session_state.best_p['ema_slow']})", line=dict(color="#ff9f43", width=1.5)))
        
        # Điểm Mua
        buy_idx = buy_s & np.isfinite(oos_close)
        fig_stock.add_trace(go.Scatter(
            x=oos_close.index[buy_idx],
            y=oos_close[buy_idx],
            mode='markers',
            name='Tín hiệu Mua',
            marker=dict(symbol='triangle-up', size=12, color='#28c76f', line=dict(width=1, color='white'))
        ))

        # Điểm Bán thông thường
        sell_idx = sell_s & np.isfinite(oos_close)
        fig_stock.add_trace(go.Scatter(
            x=oos_close.index[sell_idx],
            y=oos_close[sell_idx],
            mode='markers',
            name='Tín hiệu Bán',
            marker=dict(symbol='triangle-down', size=12, color='#ea5455', line=dict(width=1, color='white'))
        ))

        # Điểm Cắt lỗ (Stop Loss)
        fig_stock.add_trace(go.Scatter(
            x=oos_close.index[stop_hits],
            y=oos_close[stop_hits],
            mode='markers',
            name='Điểm Cắt Lỗ (Stop Loss)',
            marker=dict(symbol='x', size=12, color='#ff3366', line=dict(width=2, color='white'))
        ))

        fig_stock.update_layout(
            title=f"Đồ thị giá {selected_stock} và Tín hiệu Chiến lược (OOS)",
            xaxis_title="Thời gian",
            yaxis_title="Giá (VND)",
            template="plotly_dark",
            hovermode="x unified"
        )
        st.plotly_chart(fig_stock, use_container_width=True)

        # Biểu đồ RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=rsi_s.index, y=rsi_s.values, name="RSI", line=dict(color="#a855f7")))
        fig_rsi.add_shape(type="line", x0=rsi_s.index.min(), y0=st.session_state.best_p["rsi_lower"], x1=rsi_s.index.max(), y1=st.session_state.best_p["rsi_lower"], line=dict(color="#28c76f", dash="dash"))
        fig_rsi.add_shape(type="line", x0=rsi_s.index.min(), y0=st.session_state.best_p["rsi_upper"], x1=rsi_s.index.max(), y1=st.session_state.best_p["rsi_upper"], line=dict(color="#ea5455", dash="dash"))
        
        fig_rsi.update_layout(
            title=f"Chỉ số RSI của {selected_stock}",
            xaxis_title="Thời gian",
            yaxis_title="RSI (Wilder)",
            yaxis=dict(range=[10, 90]),
            template="plotly_dark"
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        # Các chỉ số riêng lẻ của mã này
        m_single = perf_metrics(equity_single, trades_single, initial=initial_capital, rf_daily=rf_daily, periods=trading_days)
        st.write(f"📊 **Chỉ số hiệu quả giao dịch đơn lẻ trên mã {selected_stock}:**")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Tổng lợi nhuận", f"{m_single['Total Return [%]']:.2f}%")
        with col_s2:
            st.metric("Sharpe Ratio", f"{m_single['Sharpe']:.3f}")
        with col_s3:
            st.metric("Max Drawdown", f"{m_single['Max Drawdown [%]']:.2f}%")
        with col_s4:
            st.metric("Số giao dịch thực hiện", f"{trades_single}")

    # ==========================================================================================
    # TAB 6: KIỂM ĐỊNH THỐNG KÊ
    # ==========================================================================================
    with tab_stats:
        st.subheader("Kiểm định Thống kê Ý nghĩa Chiến lược")
        st.write("Đánh giá xem kết quả vượt trội của chiến lược là thực tế hay ngẫu nhiên bằng các kiểm định thống kê toán học:")
        st.write("- **One-sample t-test**: Kiểm nghiệm xem lợi nhuận kỳ vọng của chiến lược có thực sự lớn hơn Lãi suất phi rủi ro hay không.")
        st.write("- **Wilcoxon signed-rank test**: Kiểm nghiệm xem phân phối lợi nhuận hàng ngày của Chiến lược có vượt trội hơn VN-Index và Buy & Hold hay không.")

        if not HAVE_SCIPY:
            st.warning("Ứng dụng thiếu thư viện `scipy`. Vui lòng cài đặt thư viện này để thực hiện kiểm định thống kê.")
        elif "top_stocks" not in st.session_state:
            st.warning("Vui lòng thực hiện Backtest danh mục trước.")
        else:
            top_stocks = st.session_state.top_stocks
            
            # Tính toán các chuỗi lợi nhuận
            eq_strat, _ = run_portfolio_ema_rsi(
                close, open_, top_stocks, st.session_state.best_p,
                weight_scheme="equal",
                rebalance="quarterly",
                init_perf=is_ret,
                capital=initial_capital,
                fee=fee_rate,
                oos_start=str(oos_start_date)
            )
            eq_bh_top3, _ = buy_hold_basket(close, open_, top_stocks, capital=initial_capital, fee=fee_rate, oos_start=str(oos_start_date), name="BuyHold_TopN")
            
            if "VNINDEX" in close.columns and eq_strat is not None:
                eq_vnindex, _ = buy_hold_index(close, "VNINDEX", capital=initial_capital, oos_start=str(oos_start_date))
                
                # Đồng bộ hóa ngày
                common_idx = eq_strat.index.intersection(eq_bh_top3.index).intersection(eq_vnindex.index)
                r_s = eq_strat.loc[common_idx].pct_change().dropna()
                r_b = eq_bh_top3.loc[common_idx].pct_change().dropna()
                r_i = eq_vnindex.loc[common_idx].pct_change().dropna()

                common_ret_idx = r_s.index.intersection(r_b.index).intersection(r_i.index)
                r_s = r_s.loc[common_ret_idx]
                r_b = r_b.loc[common_ret_idx]
                r_i = r_i.loc[common_ret_idx]

                # Thực hiện các kiểm định
                # 1. t-test
                t_stat, t_pval = ttest_1samp(r_s - rf_daily, 0, alternative="greater")
                
                # 2. Wilcoxon vs VNINDEX
                w_stat_i, w_pval_i = wilcoxon(r_s, r_i, alternative="greater")
                
                # 3. Wilcoxon vs Buy & Hold
                w_stat_b, w_pval_b = wilcoxon(r_s, r_b, alternative="greater")

                # Trình bày kết quả
                col_st1, col_st2 = st.columns(2)
                with col_st1:
                    st.write("🔬 **Bảng kết quả kiểm định:**")
                    
                    st.markdown(f"""
                    * **t-test (Lợi nhuận vượt trội > 0):**
                      - t-statistic: `{t_stat:.4f}`
                      - p-value: `{t_pval:.6f}`
                    
                    * **Wilcoxon test (EMA+RSI > VN-Index):**
                      - statistic: `{w_stat_i:.1f}`
                      - p-value: `{w_pval_i:.6f}`
                    
                    * **Wilcoxon test (EMA+RSI > Buy & Hold Top N):**
                      - statistic: `{w_stat_b:.1f}`
                      - p-value: `{w_pval_b:.6f}`
                    """)
                
                with col_st2:
                    st.write("📝 **Giải thích ý nghĩa:**")
                    alpha = 0.05
                    
                    st.write("**Kiểm định Lợi nhuận:**")
                    if t_pval < alpha:
                        st.success(f"✓ p-value ({t_pval:.4g}) < {alpha}: Bác bỏ giả thuyết H0. Lợi nhuận của chiến lược thực sự lớn hơn lãi suất phi rủi ro có ý nghĩa thống kê.")
                    else:
                        st.warning(f"✗ p-value ({t_pval:.4g}) >= {alpha}: Không thể bác bỏ H0. Lợi nhuận chiến lược không đủ bằng chứng lớn hơn lãi suất phi rủi ro.")

                    st.write("**So sánh với VN-Index:**")
                    if w_pval_i < alpha:
                        st.success(f"✓ p-value ({w_pval_i:.4g}) < {alpha}: Bác bỏ H0. Lợi nhuận hàng ngày của chiến lược vượt trội hơn chỉ số VN-Index có ý nghĩa thống kê.")
                    else:
                        st.warning(f"✗ p-value ({w_pval_i:.4g}) >= {alpha}: Không có ý nghĩa thống kê cho thấy chiến lược chiến thắng VN-Index.")

                    st.write("**So sánh với Buy & Hold:**")
                    if w_pval_b < alpha:
                        st.success(f"✓ p-value ({w_pval_b:.4g}) < {alpha}: Bác bỏ H0. Chiến lược năng động EMA+RSI mang lại kết quả tốt hơn việc chỉ giữ nguyên danh mục có ý nghĩa thống kê.")
                    else:
                        st.warning(f"✗ p-value ({w_pval_b:.4g}) >= {alpha}: Việc quản lý giao dịch năng động chưa chứng minh hiệu quả hơn việc chỉ giữ nguyên (Buy & Hold).")
            else:
                st.warning("Không tìm thấy dữ liệu VNINDEX để thực hiện so sánh benchmark chi tiết hoặc danh mục lỗi.")
else:
    st.error("Không tìm thấy dữ liệu HOSE_2020_2023.csv. Vui lòng tải dữ liệu lên để khởi động ứng dụng.")
