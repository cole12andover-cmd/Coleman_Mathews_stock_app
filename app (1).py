# app.py
# -------------------------------------------------------
# FDA2 Interactive Portfolio Analytics Application
# Run locally with: uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.optimize import minimize
import math
from datetime import date, timedelta

# ============================================================
# PAGE CONFIG  (must be the FIRST Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
TRADING_DAYS_PER_YEAR = 252   # Standard annualisation factor
SP500_TICKER          = "^GSPC"  # S&P 500 benchmark ticker
MIN_TICKERS           = 3        # Minimum number of valid tickers required
MAX_TICKERS           = 10       # Maximum number of valid tickers allowed
MIN_DATE_RANGE_DAYS   = 730      # Minimum date range (≈ 2 years)
DEFAULT_RISK_FREE_PCT = 2.0      # Default annualised risk-free rate (%)
MAX_MISSING_PCT       = 0.05     # Drop ticker if > 5 % of values are missing

# ============================================================
# SIDEBAR — USER INPUTS
# ============================================================
st.sidebar.header("⚙️ Portfolio Settings")

# --- Ticker entry ---
raw_ticker_input: str = st.sidebar.text_area(
    "Stock Tickers (comma-separated, 3–10)",
    value="AAPL, MSFT, GOOGL, AMZN, JPM",
    height=80,
    help="Enter between 3 and 10 ticker symbols separated by commas.",
)

# Parse tickers: split on commas, strip whitespace, uppercase, remove blanks
user_tickers_raw: list[str] = [
    t.strip().upper() for t in raw_ticker_input.split(",") if t.strip()
]

# --- Date range ---
st.sidebar.subheader("Date Range")
default_start_date: date = date.today() - timedelta(days=5 * 365)  # 5 years back
start_date: date = st.sidebar.date_input(
    "Start Date",
    value=default_start_date,
    min_value=date(1990, 1, 1),
)
end_date: date = st.sidebar.date_input(
    "End Date",
    value=date.today(),
    min_value=date(1990, 1, 1),
)

# --- Risk-free rate ---
risk_free_rate_pct: float = st.sidebar.number_input(
    "Risk-Free Rate (annualised %)",
    min_value=0.0,
    max_value=20.0,
    value=DEFAULT_RISK_FREE_PCT,
    step=0.1,
    help="Annualised risk-free rate used in Sharpe and Sortino calculations.",
)
risk_free_rate_annual: float = risk_free_rate_pct / 100.0   # Decimal form
risk_free_rate_daily:  float = risk_free_rate_annual / TRADING_DAYS_PER_YEAR  # Daily form

# --- About / Methodology (sidebar expander) ---
with st.sidebar.expander("ℹ️ About & Methodology"):
    st.markdown("""
**Data Source:** Yahoo Finance via `yfinance` (adjusted closing prices).

**Return Convention:** Simple (arithmetic) returns: `r_t = (P_t / P_{t-1}) - 1`.
Log returns are *not* used because they are not additive across assets.

**Annualisation:**
- Mean return × 252
- Std deviation × √252

**Cumulative Wealth Index:** `$10,000 × (1 + r).cumprod()`

**Portfolio Variance:** Quadratic form `wᵀ Σ w` (full covariance matrix).

**Sharpe Ratio:** `(Rp - Rf) / σp` — denominator is *total* annualised volatility.

**Sortino Ratio:** `(Rp - Rf) / σ_down` — denominator uses only returns below the daily risk-free rate.

**Optimisation:** `scipy.optimize.minimize` with no-short-selling bounds (0 ≤ wᵢ ≤ 1, Σwᵢ = 1).

**Risk Contribution:** `PRCᵢ = wᵢ · (Σw)ᵢ / σ²ₚ`

**Efficient Frontier:** Solved by minimising variance at a grid of target return levels (constrained optimisation — *not* random simulation).
""")

# ============================================================
# INPUT VALIDATION
# ============================================================
validation_errors: list[str] = []

# Validate ticker count
if len(user_tickers_raw) < MIN_TICKERS:
    validation_errors.append(
        f"Please enter at least {MIN_TICKERS} ticker symbols (you entered {len(user_tickers_raw)})."
    )
if len(user_tickers_raw) > MAX_TICKERS:
    validation_errors.append(
        f"Please enter no more than {MAX_TICKERS} ticker symbols (you entered {len(user_tickers_raw)})."
    )

# Validate date range
date_range_days: int = (end_date - start_date).days
if start_date >= end_date:
    validation_errors.append("Start date must be before end date.")
elif date_range_days < MIN_DATE_RANGE_DAYS:
    validation_errors.append(
        f"Date range must span at least 2 years. "
        f"Your range is only {date_range_days} days."
    )

if validation_errors:
    st.title("📈 Interactive Portfolio Analytics")
    for err in validation_errors:
        st.error(err)
    st.stop()

# ============================================================
# DATA DOWNLOAD & CACHING
# ============================================================
@st.cache_data(show_spinner=False, ttl=3600)
def download_adjusted_close(
    tickers: tuple[str, ...],   # Use tuple so it's hashable for caching
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.

    Returns a DataFrame with one column per ticker (including SP500).
    Returns an empty DataFrame if the download fails entirely.
    """
    all_tickers: list[str] = list(tickers) + [SP500_TICKER]
    try:
        raw: pd.DataFrame = yf.download(
            all_tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,   # Use adjusted prices (accounts for dividends / splits)
        )
    except Exception as download_error:
        st.error(f"Yahoo Finance download failed: {download_error}")
        return pd.DataFrame()

    # yfinance returns multi-level columns when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        # Keep only the 'Close' price level
        price_df: pd.DataFrame = raw["Close"].copy()
    else:
        # Single-ticker edge case (shouldn't happen here but be safe)
        price_df = raw[["Close"]].copy()
        price_df.columns = list(tickers)

    return price_df


# --- Run the download ---
with st.spinner("Downloading market data from Yahoo Finance…"):
    raw_price_df: pd.DataFrame = download_adjusted_close(
        tuple(user_tickers_raw),  # Convert to tuple for caching
        start_date,
        end_date,
    )

if raw_price_df.empty:
    st.error("No data was returned. Check your ticker symbols and date range, then try again.")
    st.stop()

# ============================================================
# HANDLE PARTIAL / MISSING DATA
# ============================================================
# Separate benchmark from user stocks
sp500_prices_raw: pd.Series = raw_price_df.get(SP500_TICKER, pd.Series(dtype=float))

# Identify which user tickers are actually in the downloaded data
downloaded_user_tickers: list[str] = [
    t for t in user_tickers_raw if t in raw_price_df.columns
]
missing_tickers: list[str] = [
    t for t in user_tickers_raw if t not in raw_price_df.columns
]

if missing_tickers:
    st.warning(
        f"The following tickers could not be found and will be excluded: "
        f"**{', '.join(missing_tickers)}**. "
        "Check that the symbols are correct."
    )

# Work only with the user stock columns (exclude benchmark column)
stock_price_df: pd.DataFrame = raw_price_df[downloaded_user_tickers].copy()

# Drop tickers with > MAX_MISSING_PCT missing values
missing_frac: pd.Series = stock_price_df.isnull().mean()
tickers_too_sparse: list[str] = missing_frac[missing_frac > MAX_MISSING_PCT].index.tolist()
if tickers_too_sparse:
    st.warning(
        f"The following tickers had more than {MAX_MISSING_PCT*100:.0f}% missing values "
        f"and were removed: **{', '.join(tickers_too_sparse)}**."
    )
    stock_price_df = stock_price_df.drop(columns=tickers_too_sparse)
    downloaded_user_tickers = [t for t in downloaded_user_tickers if t not in tickers_too_sparse]

# Re-check minimum ticker count after removals
if len(downloaded_user_tickers) < MIN_TICKERS:
    st.error(
        f"After removing invalid/sparse tickers, only {len(downloaded_user_tickers)} valid ticker(s) "
        f"remain. At least {MIN_TICKERS} are required."
    )
    st.stop()

# Truncate all series to the overlapping (common) date range — forward-fill max 1 day
stock_price_df = stock_price_df.ffill(limit=1).dropna()

# Align benchmark to the same dates
sp500_prices: pd.Series = sp500_prices_raw.reindex(stock_price_df.index).ffill(limit=1)

# Final list of valid tickers after all cleaning
valid_tickers: list[str] = list(stock_price_df.columns)
num_assets: int = len(valid_tickers)

# ============================================================
# CORE CALCULATIONS  (cached)
# ============================================================
@st.cache_data(show_spinner=False, ttl=3600)
def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple (arithmetic) daily returns from price DataFrame."""
    return prices.pct_change().dropna()


@st.cache_data(show_spinner=False, ttl=3600)
def compute_series_simple_returns(prices_series: pd.Series) -> pd.Series:
    """Compute simple daily returns for a single price series."""
    return prices_series.pct_change().dropna()


@st.cache_data(show_spinner=False, ttl=3600)
def compute_summary_statistics(
    returns_df: pd.DataFrame,
    rf_annual: float,
) -> pd.DataFrame:
    """
    Compute annualised summary statistics for each ticker.

    Columns: Annualised Return, Annualised Volatility, Skewness, Kurtosis,
             Min Daily Return, Max Daily Return, Sharpe Ratio, Sortino Ratio.
    """
    rf_daily: float = rf_annual / TRADING_DAYS_PER_YEAR
    results: dict = {}

    for ticker in returns_df.columns:
        col_returns: pd.Series = returns_df[ticker].dropna()

        ann_return:     float = col_returns.mean() * TRADING_DAYS_PER_YEAR
        ann_volatility: float = col_returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        skewness:       float = float(col_returns.skew())
        kurt:           float = float(col_returns.kurt())
        min_ret:        float = float(col_returns.min())
        max_ret:        float = float(col_returns.max())

        # Sharpe ratio: (annualised return - rf) / annualised vol
        sharpe: float = (ann_return - rf_annual) / ann_volatility if ann_volatility > 0 else np.nan

        # Sortino ratio: denominator uses only returns below the daily rf rate
        downside_returns: pd.Series = col_returns[col_returns < rf_daily] - rf_daily
        downside_dev: float = (
            math.sqrt((downside_returns ** 2).mean() * TRADING_DAYS_PER_YEAR)
            if len(downside_returns) > 0 else np.nan
        )
        sortino: float = (ann_return - rf_annual) / downside_dev if (downside_dev and downside_dev > 0) else np.nan

        results[ticker] = {
            "Ann. Return":     ann_return,
            "Ann. Volatility": ann_volatility,
            "Skewness":        skewness,
            "Kurtosis":        kurt,
            "Min Daily Ret":   min_ret,
            "Max Daily Ret":   max_ret,
            "Sharpe Ratio":    sharpe,
            "Sortino Ratio":   sortino,
        }

    return pd.DataFrame(results).T


@st.cache_data(show_spinner=False, ttl=3600)
def compute_max_drawdown(cum_wealth: pd.Series) -> float:
    """Compute maximum drawdown from a cumulative wealth (or price) series."""
    rolling_peak: pd.Series = cum_wealth.cummax()
    drawdown_series: pd.Series = (cum_wealth - rolling_peak) / rolling_peak
    return float(drawdown_series.min())


@st.cache_data(show_spinner=False, ttl=3600)
def compute_drawdown_series(cum_wealth: pd.Series) -> pd.Series:
    """Return the full drawdown-from-peak series as a percentage."""
    rolling_peak: pd.Series = cum_wealth.cummax()
    return (cum_wealth - rolling_peak) / rolling_peak * 100.0


@st.cache_data(show_spinner=False, ttl=3600)
def compute_portfolio_metrics(
    weights: np.ndarray,
    returns_df: pd.DataFrame,
    rf_annual: float,
) -> dict:
    """
    Compute annualised return, volatility, Sharpe, Sortino, and max drawdown
    for a portfolio defined by `weights` (must align with returns_df columns).
    """
    rf_daily: float = rf_annual / TRADING_DAYS_PER_YEAR

    # Portfolio daily returns
    port_daily_returns: pd.Series = returns_df.dot(weights)

    # Annualised return and volatility
    ann_return:     float = port_daily_returns.mean() * TRADING_DAYS_PER_YEAR
    ann_volatility: float = port_daily_returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe ratio
    sharpe: float = (ann_return - rf_annual) / ann_volatility if ann_volatility > 0 else np.nan

    # Sortino ratio
    downside: pd.Series = port_daily_returns[port_daily_returns < rf_daily] - rf_daily
    downside_dev: float = (
        math.sqrt((downside ** 2).mean() * TRADING_DAYS_PER_YEAR)
        if len(downside) > 0 else np.nan
    )
    sortino: float = (ann_return - rf_annual) / downside_dev if (downside_dev and downside_dev > 0) else np.nan

    # Cumulative wealth and max drawdown
    cum_wealth: pd.Series = 10_000 * (1 + port_daily_returns).cumprod()
    max_dd: float = compute_max_drawdown(cum_wealth)

    return {
        "weights":        weights,
        "ann_return":     ann_return,
        "ann_volatility": ann_volatility,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "max_drawdown":   max_dd,
        "cum_wealth":     cum_wealth,
        "daily_returns":  port_daily_returns,
    }


@st.cache_data(show_spinner=False, ttl=3600)
def optimize_portfolio(
    returns_df: pd.DataFrame,
    rf_annual: float,
    objective: str,   # "gmv" or "tangency"
) -> dict | None:
    """
    Optimise a portfolio using scipy.optimize.minimize.

    objective="gmv"      → minimise portfolio variance (Global Minimum Variance)
    objective="tangency" → minimise negative Sharpe ratio (Tangency / Max Sharpe)

    Returns a dict of results or None if optimisation fails.
    """
    n: int = returns_df.shape[1]
    rf_daily: float = rf_annual / TRADING_DAYS_PER_YEAR

    cov_matrix: np.ndarray = returns_df.cov().values * TRADING_DAYS_PER_YEAR  # Annualised cov
    mean_returns_annual: np.ndarray = returns_df.mean().values * TRADING_DAYS_PER_YEAR

    # Constraints: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Bounds: no short-selling — each weight in [0, 1]
    bounds = [(0.0, 1.0)] * n

    # Initial guess: equal weight
    w0: np.ndarray = np.array([1.0 / n] * n)

    if objective == "gmv":
        def portfolio_variance(w: np.ndarray) -> float:
            return float(w @ cov_matrix @ w)

        result = minimize(
            portfolio_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

    elif objective == "tangency":
        def neg_sharpe(w: np.ndarray) -> float:
            port_return: float = float(mean_returns_annual @ w)
            port_vol:    float = math.sqrt(float(w @ cov_matrix @ w))
            if port_vol <= 0:
                return 0.0
            return -(port_return - rf_annual) / port_vol

        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    else:
        return None

    if not result.success:
        st.warning(f"Portfolio optimisation ({objective.upper()}) did not fully converge: {result.message}")

    optimal_weights: np.ndarray = np.clip(result.x, 0.0, 1.0)
    optimal_weights /= optimal_weights.sum()  # Re-normalise to fix floating-point drift

    return compute_portfolio_metrics(optimal_weights, returns_df, rf_annual)


@st.cache_data(show_spinner=False, ttl=3600)
def compute_risk_contribution(
    weights: np.ndarray,
    returns_df: pd.DataFrame,
) -> np.ndarray:
    """
    Compute Percentage Risk Contribution (PRC) for each asset.

    PRCᵢ = wᵢ · (Σw)ᵢ / σ²ₚ
    The PRC values sum to 1.
    """
    cov_matrix: np.ndarray = returns_df.cov().values  # Daily covariance
    port_variance: float = float(weights @ cov_matrix @ weights)
    marginal_contrib: np.ndarray = cov_matrix @ weights
    prc: np.ndarray = (weights * marginal_contrib) / port_variance
    return prc


@st.cache_data(show_spinner=False, ttl=3600)
def compute_efficient_frontier(
    returns_df: pd.DataFrame,
    rf_annual: float,
    n_points: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the efficient frontier by minimising portfolio variance at
    each target return level (constrained optimisation — not simulation).

    Returns arrays of (volatilities, target_returns) along the frontier.
    """
    n: int = returns_df.shape[1]
    mean_returns_annual: np.ndarray = returns_df.mean().values * TRADING_DAYS_PER_YEAR
    cov_matrix: np.ndarray = returns_df.cov().values * TRADING_DAYS_PER_YEAR

    bounds = [(0.0, 1.0)] * n
    w0: np.ndarray = np.array([1.0 / n] * n)

    # Target return grid: from min to max individual asset return
    ret_min: float = mean_returns_annual.min()
    ret_max: float = mean_returns_annual.max()
    target_returns: np.ndarray = np.linspace(ret_min, ret_max, n_points)

    frontier_vols: list[float] = []
    frontier_rets: list[float] = []

    for target_ret in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, tr=target_ret: float(mean_returns_annual @ w) - tr},
        ]

        def port_variance(w: np.ndarray) -> float:
            return float(w @ cov_matrix @ w)

        res = minimize(
            port_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        if res.success:
            w_opt: np.ndarray = np.clip(res.x, 0.0, 1.0)
            w_opt /= w_opt.sum()
            frontier_vols.append(math.sqrt(float(w_opt @ cov_matrix @ w_opt)))
            frontier_rets.append(target_ret)

    return np.array(frontier_vols), np.array(frontier_rets)


# ============================================================
# RUN CORE CALCULATIONS
# ============================================================
with st.spinner("Computing returns and analytics…"):
    # Daily simple returns for all user stocks
    stock_returns_df: pd.DataFrame = compute_simple_returns(stock_price_df)

    # Daily returns for benchmark
    sp500_returns: pd.Series = compute_series_simple_returns(sp500_prices)

    # Align benchmark returns to stock returns dates
    sp500_returns = sp500_returns.reindex(stock_returns_df.index).dropna()

    # Summary stats for stocks + benchmark
    sp500_returns_df_aligned: pd.DataFrame = sp500_returns.rename("S&P 500").to_frame()
    all_returns_for_stats: pd.DataFrame = pd.concat(
        [stock_returns_df, sp500_returns_df_aligned], axis=1
    ).dropna()
    summary_stats_df: pd.DataFrame = compute_summary_statistics(
        all_returns_for_stats, risk_free_rate_annual
    )

    # Cumulative wealth index ($10,000 starting value)
    stock_cum_wealth_df: pd.DataFrame = 10_000 * (1 + stock_returns_df).cumprod()
    sp500_cum_wealth: pd.Series = 10_000 * (1 + sp500_returns).cumprod()

    # Equal-weight portfolio
    equal_weights: np.ndarray = np.array([1.0 / num_assets] * num_assets)
    equal_weight_metrics: dict = compute_portfolio_metrics(
        equal_weights, stock_returns_df, risk_free_rate_annual
    )

    # GMV portfolio
    gmv_metrics: dict | None = optimize_portfolio(
        stock_returns_df, risk_free_rate_annual, "gmv"
    )

    # Tangency portfolio
    tangency_metrics: dict | None = optimize_portfolio(
        stock_returns_df, risk_free_rate_annual, "tangency"
    )

    # Risk contributions
    gmv_prc: np.ndarray | None = (
        compute_risk_contribution(gmv_metrics["weights"], stock_returns_df)
        if gmv_metrics else None
    )
    tangency_prc: np.ndarray | None = (
        compute_risk_contribution(tangency_metrics["weights"], stock_returns_df)
        if tangency_metrics else None
    )

    # Efficient frontier
    frontier_vols_arr, frontier_rets_arr = compute_efficient_frontier(
        stock_returns_df, risk_free_rate_annual
    )

# ============================================================
# HELPER: FORMAT PERCENTAGE
# ============================================================
def fmt_pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def fmt_dollar(value: float) -> str:
    return f"${value:,.2f}"


# ============================================================
# MAIN TITLE & TICKER BADGE
# ============================================================
st.title("📈 Interactive Portfolio Analytics")
st.caption(
    f"Tickers: **{' | '.join(valid_tickers)}** &nbsp;·&nbsp; "
    f"Period: **{start_date}** → **{end_date}** &nbsp;·&nbsp; "
    f"Risk-Free Rate: **{risk_free_rate_pct:.1f}%**"
)

# ============================================================
# TABS
# ============================================================
tab_explore, tab_risk, tab_correlation, tab_portfolio, tab_sensitivity = st.tabs([
    "🔍 Exploratory Analysis",
    "⚠️ Risk Analysis",
    "🔗 Correlation & Covariance",
    "💼 Portfolio Construction",
    "🧪 Estimation Window Sensitivity",
])

# ============================================================
# TAB 1 — EXPLORATORY ANALYSIS
# ============================================================
with tab_explore:
    st.header("Return Computation & Exploratory Analysis")

    # --- 1. Summary statistics table ---
    st.subheader("Summary Statistics")
    display_stats: pd.DataFrame = summary_stats_df.copy()
    display_stats_fmt: pd.DataFrame = display_stats.style.format({
        "Ann. Return":     "{:.2%}",
        "Ann. Volatility": "{:.2%}",
        "Skewness":        "{:.3f}",
        "Kurtosis":        "{:.3f}",
        "Min Daily Ret":   "{:.2%}",
        "Max Daily Ret":   "{:.2%}",
        "Sharpe Ratio":    "{:.3f}",
        "Sortino Ratio":   "{:.3f}",
    })
    st.dataframe(display_stats_fmt, use_container_width=True)

    st.divider()

    # --- 2. Cumulative wealth index chart ---
    st.subheader("Cumulative Wealth Index ($10,000 start)")

    # Multi-select to show/hide individual stocks
    wealth_tickers_visible: list[str] = st.multiselect(
        "Select series to display",
        options=valid_tickers + ["S&P 500"],
        default=valid_tickers + ["S&P 500"],
        key="wealth_multiselect",
    )

    fig_wealth = go.Figure()
    for ticker in valid_tickers:
        if ticker in wealth_tickers_visible:
            fig_wealth.add_trace(go.Scatter(
                x=stock_cum_wealth_df.index,
                y=stock_cum_wealth_df[ticker],
                mode="lines",
                name=ticker,
                line=dict(width=1.8),
            ))
    if "S&P 500" in wealth_tickers_visible:
        fig_wealth.add_trace(go.Scatter(
            x=sp500_cum_wealth.index,
            y=sp500_cum_wealth.values,
            mode="lines",
            name="S&P 500",
            line=dict(width=2, dash="dash", color="black"),
        ))
    fig_wealth.update_layout(
        title="Growth of $10,000",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

    st.divider()

    # --- 3. Return distribution plot ---
    st.subheader("Return Distribution")
    col_dist_left, col_dist_right = st.columns([1, 3])

    with col_dist_left:
        dist_ticker_choice: str = st.selectbox(
            "Select stock", options=valid_tickers, key="dist_ticker"
        )
        dist_plot_type: str = st.radio(
            "Plot type",
            options=["Histogram + Normal Fit", "Q-Q Plot"],
            key="dist_plot_type",
        )

    selected_returns_for_dist: pd.Series = stock_returns_df[dist_ticker_choice].dropna()

    with col_dist_right:
        if dist_plot_type == "Histogram + Normal Fit":
            mu_dist: float = float(selected_returns_for_dist.mean())
            sigma_dist: float = float(selected_returns_for_dist.std())
            x_range_dist: np.ndarray = np.linspace(
                selected_returns_for_dist.min(), selected_returns_for_dist.max(), 300
            )
            normal_pdf_dist: np.ndarray = stats.norm.pdf(x_range_dist, mu_dist, sigma_dist)
            # Scale normal PDF to match histogram counts
            bin_width_dist: float = (selected_returns_for_dist.max() - selected_returns_for_dist.min()) / 50
            scale_factor: float = len(selected_returns_for_dist) * bin_width_dist

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=selected_returns_for_dist,
                nbinsx=50,
                name="Daily Returns",
                opacity=0.65,
                marker_color="#1f77b4",
            ))
            fig_dist.add_trace(go.Scatter(
                x=x_range_dist,
                y=normal_pdf_dist * scale_factor,
                mode="lines",
                name="Normal Fit",
                line=dict(color="red", width=2),
            ))
            fig_dist.update_layout(
                title=f"{dist_ticker_choice} — Daily Return Histogram with Normal Fit",
                xaxis_title="Daily Return",
                yaxis_title="Count",
                template="plotly_white",
                height=420,
                barmode="overlay",
            )
        else:
            # Q-Q plot using scipy.stats.probplot
            qq_theoretical, qq_sample = stats.probplot(selected_returns_for_dist, dist="norm", fit=False)
            qq_line_x: np.ndarray = np.array([qq_theoretical[0], qq_theoretical[-1]])
            qq_line_y: np.ndarray = qq_line_x  # 45-degree reference line for standard normal

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(
                x=qq_theoretical,
                y=qq_sample,
                mode="markers",
                name="Return Quantiles",
                marker=dict(color="#1f77b4", size=4, opacity=0.7),
            ))
            fig_dist.add_trace(go.Scatter(
                x=qq_line_x,
                y=qq_line_y,
                mode="lines",
                name="Normal Reference",
                line=dict(color="red", width=2, dash="dash"),
            ))
            fig_dist.update_layout(
                title=f"{dist_ticker_choice} — Q-Q Plot (vs. Normal Distribution)",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                template="plotly_white",
                height=420,
            )
            st.caption(
                "**Q-Q Plot Interpretation:** Points that deviate from the red reference line "
                "indicate departures from normality. Pronounced curves at the tails indicate "
                "fat tails (leptokurtosis), which are common in stock returns."
            )

        st.plotly_chart(fig_dist, use_container_width=True)

# ============================================================
# TAB 2 — RISK ANALYSIS
# ============================================================
with tab_risk:
    st.header("Risk Analysis")

    # --- 1. Rolling volatility ---
    st.subheader("Rolling Annualised Volatility")
    rolling_window_days: int = st.select_slider(
        "Rolling window (days)",
        options=[30, 60, 90, 120],
        value=60,
        key="rolling_vol_window",
    )
    fig_rolling_vol = go.Figure()
    for ticker in valid_tickers:
        rolling_vol_series: pd.Series = (
            stock_returns_df[ticker]
            .rolling(window=rolling_window_days)
            .std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        )
        fig_rolling_vol.add_trace(go.Scatter(
            x=rolling_vol_series.index,
            y=rolling_vol_series.values,
            mode="lines",
            name=ticker,
            line=dict(width=1.5),
        ))
    fig_rolling_vol.update_layout(
        title=f"Rolling {rolling_window_days}-Day Annualised Volatility",
        xaxis_title="Date",
        yaxis_title="Annualised Volatility",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_rolling_vol, use_container_width=True)
    st.caption("Note: The first (window − 1) observations are NaN and are not plotted — this is expected.")

    st.divider()

    # --- 2. Drawdown analysis ---
    st.subheader("Drawdown Analysis")
    drawdown_ticker_choice: str = st.selectbox(
        "Select stock for drawdown", options=valid_tickers, key="drawdown_ticker"
    )
    dd_cum_wealth: pd.Series = stock_cum_wealth_df[drawdown_ticker_choice]
    dd_series: pd.Series = compute_drawdown_series(dd_cum_wealth)
    max_dd_value: float = dd_series.min() / 100.0  # Convert pct back to decimal for metric display

    st.metric(
        label=f"{drawdown_ticker_choice} Maximum Drawdown",
        value=fmt_pct(max_dd_value),
        delta=None,
    )

    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(go.Scatter(
        x=dd_series.index,
        y=dd_series.values,
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="crimson", width=1.5),
        fillcolor="rgba(220,20,60,0.15)",
    ))
    fig_drawdown.update_layout(
        title=f"{drawdown_ticker_choice} — Drawdown from Running Peak",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis_ticksuffix="%",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_drawdown, use_container_width=True)

    st.divider()

    # --- 3. Risk-adjusted metrics table ---
    st.subheader("Risk-Adjusted Metrics")
    st.dataframe(
        summary_stats_df[["Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Sortino Ratio"]]
        .style.format({
            "Ann. Return":     "{:.2%}",
            "Ann. Volatility": "{:.2%}",
            "Sharpe Ratio":    "{:.3f}",
            "Sortino Ratio":   "{:.3f}",
        }),
        use_container_width=True,
    )

# ============================================================
# TAB 3 — CORRELATION & COVARIANCE
# ============================================================
with tab_correlation:
    st.header("Correlation & Covariance Analysis")

    # --- 1. Correlation heatmap ---
    st.subheader("Pairwise Correlation Heatmap")
    corr_matrix: pd.DataFrame = stock_returns_df.corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
    ))
    fig_corr.update_layout(
        title="Pairwise Correlation of Daily Returns",
        xaxis_title="Ticker",
        yaxis_title="Ticker",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # --- 2. Rolling correlation ---
    st.subheader("Rolling Pairwise Correlation")
    col_roll_a, col_roll_b, col_roll_w = st.columns(3)
    with col_roll_a:
        roll_corr_ticker_a: str = st.selectbox(
            "Stock A", options=valid_tickers, index=0, key="roll_corr_a"
        )
    with col_roll_b:
        other_tickers_for_b: list[str] = [t for t in valid_tickers if t != roll_corr_ticker_a]
        roll_corr_ticker_b: str = st.selectbox(
            "Stock B",
            options=other_tickers_for_b,
            index=0 if other_tickers_for_b else 0,
            key="roll_corr_b",
        )
    with col_roll_w:
        roll_corr_window: int = st.selectbox(
            "Window (days)", options=[30, 60, 90, 120], index=1, key="roll_corr_window"
        )

    rolling_corr_series: pd.Series = (
        stock_returns_df[roll_corr_ticker_a]
        .rolling(window=roll_corr_window)
        .corr(stock_returns_df[roll_corr_ticker_b])
    )
    fig_roll_corr = go.Figure()
    fig_roll_corr.add_trace(go.Scatter(
        x=rolling_corr_series.index,
        y=rolling_corr_series.values,
        mode="lines",
        name=f"Corr({roll_corr_ticker_a}, {roll_corr_ticker_b})",
        line=dict(color="#2ca02c", width=1.8),
    ))
    fig_roll_corr.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.6)
    fig_roll_corr.update_layout(
        title=f"Rolling {roll_corr_window}-Day Correlation: {roll_corr_ticker_a} vs {roll_corr_ticker_b}",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.1, 1.1]),
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_roll_corr, use_container_width=True)

    st.divider()

    # --- 3. Covariance matrix (in expander) ---
    with st.expander("📊 View Covariance Matrix (daily returns)"):
        cov_matrix_display: pd.DataFrame = stock_returns_df.cov()
        st.dataframe(
            cov_matrix_display.style.format("{:.8f}"),
            use_container_width=True,
        )
        st.caption(
            "Covariance values are daily. Multiply by 252 to annualise."
        )

# ============================================================
# TAB 4 — PORTFOLIO CONSTRUCTION
# ============================================================
with tab_portfolio:
    st.header("Portfolio Construction & Optimisation")

    # ---- Helper: format a metrics row ----
    def display_portfolio_metrics_row(metrics_dict: dict, label: str) -> None:
        """Render key portfolio metrics as st.metric widgets in a row."""
        cols = st.columns(5)
        cols[0].metric(f"{label} — Ann. Return",    fmt_pct(metrics_dict["ann_return"]))
        cols[1].metric(f"{label} — Ann. Volatility", fmt_pct(metrics_dict["ann_volatility"]))
        cols[2].metric(f"{label} — Sharpe",          f"{metrics_dict['sharpe']:.3f}")
        cols[3].metric(f"{label} — Sortino",          f"{metrics_dict['sortino']:.3f}")
        cols[4].metric(f"{label} — Max Drawdown",    fmt_pct(metrics_dict["max_drawdown"]))

    # ============================================================
    # 4A. EQUAL-WEIGHT PORTFOLIO
    # ============================================================
    st.subheader("1/N Equal-Weight Portfolio")
    display_portfolio_metrics_row(equal_weight_metrics, "Equal-Weight")

    fig_ew_weights = go.Figure(go.Bar(
        x=valid_tickers,
        y=equal_weights * 100,
        marker_color="#1f77b4",
        text=[f"{w*100:.1f}%" for w in equal_weights],
        textposition="outside",
    ))
    fig_ew_weights.update_layout(
        title="Equal-Weight Portfolio Weights",
        xaxis_title="Ticker",
        yaxis_title="Weight (%)",
        template="plotly_white",
        height=350,
    )
    st.plotly_chart(fig_ew_weights, use_container_width=True)

    st.divider()

    # ============================================================
    # 4B. OPTIMISED PORTFOLIOS (GMV & TANGENCY)
    # ============================================================
    st.subheader("Optimised Portfolios")

    opt_col_gmv, opt_col_tang = st.columns(2)

    with opt_col_gmv:
        st.markdown("#### Global Minimum Variance (GMV)")
        if gmv_metrics:
            display_portfolio_metrics_row(gmv_metrics, "GMV")
            gmv_weights_pct: list[float] = [w * 100 for w in gmv_metrics["weights"]]
            fig_gmv_w = go.Figure(go.Bar(
                x=valid_tickers,
                y=gmv_weights_pct,
                marker_color="#ff7f0e",
                text=[f"{w:.1f}%" for w in gmv_weights_pct],
                textposition="outside",
            ))
            fig_gmv_w.update_layout(
                title="GMV Portfolio Weights",
                xaxis_title="Ticker",
                yaxis_title="Weight (%)",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_gmv_w, use_container_width=True)
        else:
            st.error("GMV optimisation failed.")

    with opt_col_tang:
        st.markdown("#### Maximum Sharpe Ratio (Tangency)")
        if tangency_metrics:
            display_portfolio_metrics_row(tangency_metrics, "Tangency")
            tang_weights_pct: list[float] = [w * 100 for w in tangency_metrics["weights"]]
            fig_tang_w = go.Figure(go.Bar(
                x=valid_tickers,
                y=tang_weights_pct,
                marker_color="#2ca02c",
                text=[f"{w:.1f}%" for w in tang_weights_pct],
                textposition="outside",
            ))
            fig_tang_w.update_layout(
                title="Tangency Portfolio Weights",
                xaxis_title="Ticker",
                yaxis_title="Weight (%)",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_tang_w, use_container_width=True)
        else:
            st.error("Tangency optimisation failed.")

    st.divider()

    # ============================================================
    # 4C. RISK CONTRIBUTION
    # ============================================================
    st.subheader("Percentage Risk Contribution (PRC)")
    st.info(
        "**What is Risk Contribution?** "
        "The PRC for each asset measures what fraction of the total portfolio variance it is responsible for. "
        "A stock with 10% weight but 25% PRC is a disproportionate source of portfolio volatility — "
        "it contributes far more risk than its capital allocation suggests. "
        "Ideal risk-balanced portfolios have PRC values that closely match portfolio weights."
    )

    prc_col_gmv, prc_col_tang = st.columns(2)

    with prc_col_gmv:
        if gmv_prc is not None and gmv_metrics is not None:
            fig_prc_gmv = go.Figure()
            fig_prc_gmv.add_trace(go.Bar(
                name="Weight",
                x=valid_tickers,
                y=[w * 100 for w in gmv_metrics["weights"]],
                marker_color="#ff7f0e",
                opacity=0.7,
            ))
            fig_prc_gmv.add_trace(go.Bar(
                name="Risk Contribution",
                x=valid_tickers,
                y=[p * 100 for p in gmv_prc],
                marker_color="#d62728",
                opacity=0.85,
            ))
            fig_prc_gmv.update_layout(
                title="GMV: Weight vs Risk Contribution",
                xaxis_title="Ticker",
                yaxis_title="%",
                barmode="group",
                template="plotly_white",
                height=380,
            )
            st.plotly_chart(fig_prc_gmv, use_container_width=True)

    with prc_col_tang:
        if tangency_prc is not None and tangency_metrics is not None:
            fig_prc_tang = go.Figure()
            fig_prc_tang.add_trace(go.Bar(
                name="Weight",
                x=valid_tickers,
                y=[w * 100 for w in tangency_metrics["weights"]],
                marker_color="#2ca02c",
                opacity=0.7,
            ))
            fig_prc_tang.add_trace(go.Bar(
                name="Risk Contribution",
                x=valid_tickers,
                y=[p * 100 for p in tangency_prc],
                marker_color="#d62728",
                opacity=0.85,
            ))
            fig_prc_tang.update_layout(
                title="Tangency: Weight vs Risk Contribution",
                xaxis_title="Ticker",
                yaxis_title="%",
                barmode="group",
                template="plotly_white",
                height=380,
            )
            st.plotly_chart(fig_prc_tang, use_container_width=True)

    st.divider()

    # ============================================================
    # 4D. CUSTOM PORTFOLIO
    # ============================================================
    st.subheader("Custom Portfolio Builder")
    st.caption(
        "Adjust the sliders below. Weights are automatically normalised to sum to 1. "
        "Metrics update in real time."
    )

    # One slider per ticker (raw values; we normalise)
    custom_raw_weights: dict[str, float] = {}
    slider_cols = st.columns(num_assets)
    for i, ticker in enumerate(valid_tickers):
        with slider_cols[i]:
            custom_raw_weights[ticker] = st.slider(
                f"{ticker}",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key=f"custom_slider_{ticker}",
            )

    raw_total: float = sum(custom_raw_weights.values())
    if raw_total == 0:
        st.warning("All slider values are zero. Using equal weights.")
        custom_weights_normalised: np.ndarray = equal_weights.copy()
    else:
        custom_weights_normalised = np.array(
            [custom_raw_weights[t] / raw_total for t in valid_tickers]
        )

    # Display normalised weights
    norm_weight_labels: str = "  |  ".join(
        [f"**{t}**: {w*100:.1f}%" for t, w in zip(valid_tickers, custom_weights_normalised)]
    )
    st.markdown(f"**Normalised Weights:** {norm_weight_labels}")

    custom_metrics: dict = compute_portfolio_metrics(
        custom_weights_normalised, stock_returns_df, risk_free_rate_annual
    )
    display_portfolio_metrics_row(custom_metrics, "Custom")

    st.divider()

    # ============================================================
    # 4E. EFFICIENT FRONTIER
    # ============================================================
    st.subheader("Efficient Frontier")
    st.info(
        "**Efficient Frontier:** Each point on the curve represents the minimum-variance portfolio "
        "achievable at that return level (no-short-selling constraints). Portfolios below/right of the "
        "curve are suboptimal. "
        "**Capital Allocation Line (CAL):** The straight line from the risk-free rate through the "
        "tangency (max-Sharpe) portfolio. Any point on the CAL is achievable by combining the tangency "
        "portfolio with risk-free lending/borrowing."
    )

    # Individual stock coordinates
    stock_ann_returns:  np.ndarray = stock_returns_df.mean().values * TRADING_DAYS_PER_YEAR
    stock_ann_vols:     np.ndarray = stock_returns_df.std().values * math.sqrt(TRADING_DAYS_PER_YEAR)

    # Benchmark coordinates
    sp500_ann_return:   float = float(sp500_returns.mean()) * TRADING_DAYS_PER_YEAR
    sp500_ann_vol:      float = float(sp500_returns.std()) * math.sqrt(TRADING_DAYS_PER_YEAR)

    fig_ef = go.Figure()

    # Efficient frontier curve
    if len(frontier_vols_arr) > 0:
        fig_ef.add_trace(go.Scatter(
            x=frontier_vols_arr,
            y=frontier_rets_arr,
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="#1f77b4", width=3),
        ))

        # Capital Allocation Line: from (0, rf) through tangency
        if tangency_metrics:
            tang_vol:  float = tangency_metrics["ann_volatility"]
            tang_ret:  float = tangency_metrics["ann_return"]
            cal_slope: float = (tang_ret - risk_free_rate_annual) / tang_vol if tang_vol > 0 else 0
            cal_x:     np.ndarray = np.linspace(0, max(frontier_vols_arr) * 1.2, 100)
            cal_y:     np.ndarray = risk_free_rate_annual + cal_slope * cal_x
            fig_ef.add_trace(go.Scatter(
                x=cal_x,
                y=cal_y,
                mode="lines",
                name="Capital Allocation Line",
                line=dict(color="goldenrod", width=2, dash="dash"),
            ))

    # Individual stocks
    fig_ef.add_trace(go.Scatter(
        x=stock_ann_vols,
        y=stock_ann_returns,
        mode="markers+text",
        name="Individual Stocks",
        marker=dict(size=10, color="grey", symbol="circle"),
        text=valid_tickers,
        textposition="top center",
    ))

    # Benchmark
    fig_ef.add_trace(go.Scatter(
        x=[sp500_ann_vol],
        y=[sp500_ann_return],
        mode="markers+text",
        name="S&P 500",
        marker=dict(size=12, color="black", symbol="diamond"),
        text=["S&P 500"],
        textposition="top center",
    ))

    # Equal-weight portfolio
    fig_ef.add_trace(go.Scatter(
        x=[equal_weight_metrics["ann_volatility"]],
        y=[equal_weight_metrics["ann_return"]],
        mode="markers+text",
        name="Equal-Weight",
        marker=dict(size=14, color="#1f77b4", symbol="star"),
        text=["EW"],
        textposition="top center",
    ))

    # GMV portfolio
    if gmv_metrics:
        fig_ef.add_trace(go.Scatter(
            x=[gmv_metrics["ann_volatility"]],
            y=[gmv_metrics["ann_return"]],
            mode="markers+text",
            name="GMV",
            marker=dict(size=14, color="#ff7f0e", symbol="star"),
            text=["GMV"],
            textposition="top center",
        ))

    # Tangency portfolio
    if tangency_metrics:
        fig_ef.add_trace(go.Scatter(
            x=[tangency_metrics["ann_volatility"]],
            y=[tangency_metrics["ann_return"]],
            mode="markers+text",
            name="Tangency",
            marker=dict(size=14, color="#2ca02c", symbol="star"),
            text=["Tangency"],
            textposition="top center",
        ))

    # Custom portfolio
    fig_ef.add_trace(go.Scatter(
        x=[custom_metrics["ann_volatility"]],
        y=[custom_metrics["ann_return"]],
        mode="markers+text",
        name="Custom",
        marker=dict(size=14, color="purple", symbol="star"),
        text=["Custom"],
        textposition="top center",
    ))

    fig_ef.update_layout(
        title="Efficient Frontier with Portfolio Comparison",
        xaxis_title="Annualised Volatility (σ)",
        yaxis_title="Annualised Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=580,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    st.divider()

    # ============================================================
    # 4F. PORTFOLIO COMPARISON
    # ============================================================
    st.subheader("Portfolio Comparison")

    # Cumulative wealth chart — all portfolios + benchmark
    fig_port_compare = go.Figure()

    port_comparison_data: dict[str, dict] = {
        "Equal-Weight": equal_weight_metrics,
        "GMV":          gmv_metrics if gmv_metrics else {},
        "Tangency":     tangency_metrics if tangency_metrics else {},
        "Custom":       custom_metrics,
    }
    port_colours: dict[str, str] = {
        "Equal-Weight": "#1f77b4",
        "GMV":          "#ff7f0e",
        "Tangency":     "#2ca02c",
        "Custom":       "purple",
    }

    for port_name, port_data in port_comparison_data.items():
        if port_data and "cum_wealth" in port_data:
            fig_port_compare.add_trace(go.Scatter(
                x=port_data["cum_wealth"].index,
                y=port_data["cum_wealth"].values,
                mode="lines",
                name=port_name,
                line=dict(color=port_colours[port_name], width=2),
            ))

    fig_port_compare.add_trace(go.Scatter(
        x=sp500_cum_wealth.index,
        y=sp500_cum_wealth.values,
        mode="lines",
        name="S&P 500",
        line=dict(color="black", width=2, dash="dash"),
    ))
    fig_port_compare.update_layout(
        title="Portfolio Cumulative Wealth Index ($10,000 start)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_port_compare, use_container_width=True)

    # Summary comparison table
    st.subheader("Portfolio Summary Comparison Table")

    # Build comparison rows
    comparison_rows: dict[str, dict] = {}

    for port_name, port_data in port_comparison_data.items():
        if port_data and "ann_return" in port_data:
            comparison_rows[port_name] = {
                "Ann. Return":     port_data["ann_return"],
                "Ann. Volatility": port_data["ann_volatility"],
                "Sharpe Ratio":    port_data["sharpe"],
                "Sortino Ratio":   port_data["sortino"],
                "Max Drawdown":    port_data["max_drawdown"],
            }

    # Add S&P 500 benchmark row
    sp500_port_metrics_dict: dict = compute_portfolio_metrics(
        np.array([1.0]),  # single-asset "portfolio"
        sp500_returns.rename("SP500").to_frame(),
        risk_free_rate_annual,
    )
    comparison_rows["S&P 500"] = {
        "Ann. Return":     sp500_port_metrics_dict["ann_return"],
        "Ann. Volatility": sp500_port_metrics_dict["ann_volatility"],
        "Sharpe Ratio":    sp500_port_metrics_dict["sharpe"],
        "Sortino Ratio":   sp500_port_metrics_dict["sortino"],
        "Max Drawdown":    sp500_port_metrics_dict["max_drawdown"],
    }

    comparison_df: pd.DataFrame = pd.DataFrame(comparison_rows).T
    st.dataframe(
        comparison_df.style.format({
            "Ann. Return":     "{:.2%}",
            "Ann. Volatility": "{:.2%}",
            "Sharpe Ratio":    "{:.3f}",
            "Sortino Ratio":   "{:.3f}",
            "Max Drawdown":    "{:.2%}",
        }),
        use_container_width=True,
    )

# ============================================================
# TAB 5 — ESTIMATION WINDOW SENSITIVITY
# ============================================================
with tab_sensitivity:
    st.header("Estimation Window Sensitivity Analysis")
    st.info(
        "**Why does this matter?** Mean-variance optimisation is highly sensitive to its inputs. "
        "Small changes in estimated returns or covariances can produce dramatically different "
        "portfolio weights. The tables below show how the GMV and Tangency portfolios change "
        "depending on how many years of history are used to estimate inputs — highlighting that "
        "past optimisation results are only as stable as the data that produced them."
    )

    # Determine available lookback options based on date range
    total_years_available: float = date_range_days / 365.25
    all_lookback_options: dict[str, int] = {
        "1 Year":     252,
        "3 Years":    756,
        "5 Years":    1260,
        "Full Sample": len(stock_returns_df),
    }

    # Only offer lookbacks that fit within the user's data
    valid_lookback_options: dict[str, int] = {
        label: days
        for label, days in all_lookback_options.items()
        if days <= len(stock_returns_df)
    }

    if len(valid_lookback_options) < 2:
        st.warning(
            "Not enough data for multiple lookback windows. "
            "Extend your date range to at least 3 years for a meaningful comparison."
        )
    else:
        selected_lookback_labels: list[str] = st.multiselect(
            "Select lookback windows to compare",
            options=list(valid_lookback_options.keys()),
            default=list(valid_lookback_options.keys()),
            key="lookback_multiselect",
        )

        if not selected_lookback_labels:
            st.warning("Select at least one lookback window.")
        else:
            # Build results table for each lookback
            sensitivity_rows_gmv:     list[dict] = []
            sensitivity_rows_tangency: list[dict] = []

            with st.spinner("Running sensitivity analysis…"):
                for label in selected_lookback_labels:
                    lookback_days: int = valid_lookback_options[label]
                    # Use the most recent `lookback_days` rows
                    returns_window: pd.DataFrame = stock_returns_df.iloc[-lookback_days:]

                    gmv_sens:  dict | None = optimize_portfolio(
                        returns_window, risk_free_rate_annual, "gmv"
                    )
                    tang_sens: dict | None = optimize_portfolio(
                        returns_window, risk_free_rate_annual, "tangency"
                    )

                    # Build row for GMV
                    gmv_row: dict = {"Window": label}
                    if gmv_sens:
                        for i, ticker in enumerate(valid_tickers):
                            gmv_row[f"{ticker} Wt"] = gmv_sens["weights"][i]
                        gmv_row["Ann. Return"] = gmv_sens["ann_return"]
                        gmv_row["Ann. Volatility"] = gmv_sens["ann_volatility"]
                    sensitivity_rows_gmv.append(gmv_row)

                    # Build row for Tangency
                    tang_row: dict = {"Window": label}
                    if tang_sens:
                        for i, ticker in enumerate(valid_tickers):
                            tang_row[f"{ticker} Wt"] = tang_sens["weights"][i]
                        tang_row["Ann. Return"] = tang_sens["ann_return"]
                        tang_row["Ann. Volatility"] = tang_sens["ann_volatility"]
                        tang_row["Sharpe Ratio"] = tang_sens["sharpe"]
                    sensitivity_rows_tangency.append(tang_row)

            # Display GMV sensitivity
            st.subheader("GMV Portfolio — Sensitivity to Lookback Window")
            if sensitivity_rows_gmv:
                gmv_sens_df: pd.DataFrame = pd.DataFrame(sensitivity_rows_gmv).set_index("Window")
                weight_cols_gmv: list[str] = [c for c in gmv_sens_df.columns if "Wt" in c]
                metric_cols_gmv: list[str] = ["Ann. Return", "Ann. Volatility"]
                fmt_dict_gmv: dict = {c: "{:.1%}" for c in weight_cols_gmv + metric_cols_gmv}
                st.dataframe(
                    gmv_sens_df.style.format(fmt_dict_gmv, na_rep="N/A"),
                    use_container_width=True,
                )

                # Grouped bar chart of GMV weights
                if len(selected_lookback_labels) > 1:
                    fig_gmv_sens = go.Figure()
                    for ticker in valid_tickers:
                        wt_col: str = f"{ticker} Wt"
                        if wt_col in gmv_sens_df.columns:
                            fig_gmv_sens.add_trace(go.Bar(
                                name=ticker,
                                x=gmv_sens_df.index.tolist(),
                                y=(gmv_sens_df[wt_col] * 100).tolist(),
                            ))
                    fig_gmv_sens.update_layout(
                        title="GMV Weights Across Lookback Windows",
                        xaxis_title="Lookback Window",
                        yaxis_title="Weight (%)",
                        barmode="group",
                        template="plotly_white",
                        height=380,
                    )
                    st.plotly_chart(fig_gmv_sens, use_container_width=True)

            st.divider()

            # Display Tangency sensitivity
            st.subheader("Tangency Portfolio — Sensitivity to Lookback Window")
            if sensitivity_rows_tangency:
                tang_sens_df: pd.DataFrame = pd.DataFrame(sensitivity_rows_tangency).set_index("Window")
                weight_cols_tang: list[str] = [c for c in tang_sens_df.columns if "Wt" in c]
                metric_cols_tang: list[str] = [
                    c for c in ["Ann. Return", "Ann. Volatility", "Sharpe Ratio"]
                    if c in tang_sens_df.columns
                ]
                pct_cols_tang: list[str] = [c for c in weight_cols_tang + ["Ann. Return", "Ann. Volatility"] if c in tang_sens_df.columns]
                fmt_dict_tang: dict = {c: "{:.1%}" for c in pct_cols_tang}
                if "Sharpe Ratio" in tang_sens_df.columns:
                    fmt_dict_tang["Sharpe Ratio"] = "{:.3f}"
                st.dataframe(
                    tang_sens_df.style.format(fmt_dict_tang, na_rep="N/A"),
                    use_container_width=True,
                )

                if len(selected_lookback_labels) > 1:
                    fig_tang_sens = go.Figure()
                    for ticker in valid_tickers:
                        wt_col = f"{ticker} Wt"
                        if wt_col in tang_sens_df.columns:
                            fig_tang_sens.add_trace(go.Bar(
                                name=ticker,
                                x=tang_sens_df.index.tolist(),
                                y=(tang_sens_df[wt_col] * 100).tolist(),
                            ))
                    fig_tang_sens.update_layout(
                        title="Tangency Weights Across Lookback Windows",
                        xaxis_title="Lookback Window",
                        yaxis_title="Weight (%)",
                        barmode="group",
                        template="plotly_white",
                        height=380,
                    )
                    st.plotly_chart(fig_tang_sens, use_container_width=True)
