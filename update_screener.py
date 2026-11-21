from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
import os
import urllib.parse
import feedparser
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yfinance as yf

# Curated Nifty 100 + Key Watchlist Tickers
UNIVERSE_TICKERS = list(
    set([
        "COCHINSHIP.NS",
        "INFY.NS",
        "ITC.NS",
        "LGEINDIA.NS",
        "RELIANCE.NS",
        "SHRIRAMFIN.NS",
        "IDEA.NS",
        "LT.NS",
        "BEL.NS",
        "HDFCBANK.NS",
        "ETERNAL.NS",
        "HAL.NS",
        "DLF.NS",
        "DRREDDY.NS",
        "TMPV.NS",
        "TCS.NS",
        "ICICIBANK.NS",
        "BHARTIARTL.NS",
        "SBIN.NS",
        "LICI.NS",
        "HINDUNILVR.NS",
        "BAJFINANCE.NS",
        "SUNPHARMA.NS",
        "M&M.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "MARUTI.NS",
        "ULTRACEMCO.NS",
        "NTPC.NS",
        "TITAN.NS",
        "ADANIENT.NS",
        "ONGC.NS",
        "TATAMOTORS.NS",
        "POWERGRID.NS",
        "ADANIPORTS.NS",
        "COALINDIA.NS",
        "BAJAJFINSV.NS",
        "ASIANPAINT.NS",
        "JSWSTEEL.NS",
        "TATASTEEL.NS",
        "TRENT.NS",
        "ZOMATO.NS",
        "SIEMENS.NS",
        "ABB.NS",
        "PIDILITIND.NS",
        "GRASIM.NS",
        "INDIGO.NS",
        "SBILIFE.NS",
        "HDFCLIFE.NS",
        "TECHM.NS",
        "WIPRO.NS",
        "HINDALCO.NS",
        "CIPLA.NS",
        "EICHERMOT.NS",
        "DIVISLAB.NS",
        "BPCL.NS",
        "BRITANNIA.NS",
        "APOLLOHOSP.NS",
        "TATACONSUM.NS",
        "HEROMOTOCO.NS",
        "BAJAJ-AUTO.NS",
        "CHOLAFIN.NS",
        "BANKBARODA.NS",
        "PNB.NS",
        "IOB.NS",
        "CANBK.NS",
        "UNIONBANK.NS",
        "RECLTD.NS",
        "PFC.NS",
        "IRFC.NS",
        "VBL.NS",
        "GODREJCP.NS",
        "DABUR.NS",
        "MARICO.NS",
        "COLPAL.NS",
        "HAVELLS.NS",
        "AMBUJACEM.NS",
        "SHREECEM.NS",
        "GAIL.NS",
        "IOC.NS",
        "VEDL.NS",
        "JINDALSTEL.NS",
        "TVSMOTOR.NS",
        "MOTHERSON.NS",
        "BOSCHLTD.NS",
        "ICICIGI.NS",
        "ICICIPRULI.NS",
        "SBICARD.NS",
        "NAUKRI.NS",
        "UNITDSPR.NS",
        "TORNTPHARM.NS",
        "ZYDUSLIFE.NS",
        "LUPIN.NS",
        "AUROPHARMA.NS",
        "MANKIND.NS",
        "CGPOWER.NS",
        "POLYCAB.NS",
        "CUMMINSIND.NS",
        "ASTRAL.NS",
        "INDHOTEL.NS",
        "LODHA.NS",
        "GODREJPROP.NS",
        "OBEROIRLTY.NS",
    ])
)

SPAM_DOMAINS = {
    "marketsmojo.com",
    "tipranks.com",
    "defenseworld.net",
    "marketbeat.com",
    "simplywall.st",
    "stockinvest.us",
    "walletinvestor.com",
}


def fetch_clean_headlines(company_name, symbol):
  query = urllib.parse.quote(f'"{company_name}" OR "{symbol}" stock India')
  url = (
      f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
  )
  feed = feedparser.parse(url)
  cutoff = datetime.now(timezone.utc) - timedelta(days=7)
  clean_titles = []

  for entry in feed.entries[:25]:
    if hasattr(entry, "published"):
      try:
        if parsedate_to_datetime(entry.published) < cutoff:
          continue
      except Exception:
        pass
    source_href = (
        entry.get("source", {}).get("href", "").lower()
        if isinstance(entry.get("source"), dict)
        else ""
    )
    if any(spam in source_href for spam in SPAM_DOMAINS):
      continue
    title = entry.title.rsplit(" - ", 1)[0].strip()
    if len(title) < 15:
      continue
    if not any(
        SequenceMatcher(None, title.lower(), existing.lower()).ratio() > 0.75
        for existing in clean_titles
    ):
      clean_titles.append(title)
  return clean_titles[:10]


def score_headlines_finbert(ticker_headlines_map):
  print("Loading ProsusAI/finbert model...")
  model_name = "ProsusAI/finbert"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)
  model.eval()

  flat_texts = []
  ticker_slices = {}
  idx = 0
  for ticker, titles in ticker_headlines_map.items():
    if len(titles) >= 3:
      ticker_slices[ticker] = (idx, idx + len(titles))
      flat_texts.extend(titles)
      idx += len(titles)

  scores = {}
  if not flat_texts:
    return scores

  net_polarities = []
  with torch.no_grad():
    for i in range(0, len(flat_texts), 32):
      batch = flat_texts[i : i + 32]
      inputs = tokenizer(
          batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
      )
      outputs = model(**inputs)
      probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
      for p in probs:
        net_polarities.append(float(p[0] - p[1]))

  for ticker, (start, end) in ticker_slices.items():
    scores[ticker] = round(float(np.mean(net_polarities[start:end])), 3)
  return scores


def fetch_universe_data():
  print("Downloading bulk price & volume history...")
  all_tickers = UNIVERSE_TICKERS + ["^NSEI"]
  hist_data = yf.download(
      all_tickers, period="8mo", group_by="ticker", progress=False, threads=True
  )

  records = []
  headlines_map = {}

  for ticker in UNIVERSE_TICKERS:
    try:
      if ticker not in hist_data.columns.get_level_values(0):
        continue
      df_t = hist_data[ticker].dropna(subset=["Close", "Volume"])
      if len(df_t) < 60:
        continue

      current_price = float(df_t["Close"].iloc[-1])
      prev_close = float(df_t["Close"].iloc[-2])
      change_pct = ((current_price - prev_close) / prev_close) * 100

      avg_turnover_cr = (
          df_t["Close"].iloc[-20:] * df_t["Volume"].iloc[-20:]
      ).mean() / 1e7
      if avg_turnover_cr < 25.0:
        continue

      if len(df_t) >= 130:
        price_6m_ago = float(df_t["Close"].iloc[-130])
        price_1m_ago = float(df_t["Close"].iloc[-21])
        mom_6m_1m = ((price_1m_ago - price_6m_ago) / price_6m_ago) * 100
      else:
        mom_6m_1m = np.nan

      info = yf.Ticker(ticker).info
      sector = info.get("sector", "General")
      short_name = info.get("shortName", ticker.replace(".NS", ""))
      pe = info.get("trailingPE", None)
      pb = info.get("priceToBook", None)
      roe = info.get("returnOnEquity", None)
      de = info.get("debtToEquity", None)
      rev_growth = info.get("revenueGrowth", None)
      eps_growth = info.get("earningsGrowth", None)
      eps = info.get("trailingEps", None)

      is_financial = sector == "Financial Services"
      if eps is not None and eps <= 0:
        val_tag = "NM (Loss-Making)"
        val_metric = np.nan
      elif pb is not None and pb <= 0:
        val_tag = "NM (Neg. Equity)"
        val_metric = np.nan
      elif not is_financial and pe is not None and pe > 100:
        val_tag = "NM (P/E > 100)"
        val_metric = np.nan
      else:
        val_tag = "Valid"
        val_metric = pb if is_financial else pe

      symbol = ticker.replace(".NS", "")
      clean_titles = fetch_clean_headlines(short_name, symbol)
      headlines_map[symbol] = clean_titles

      records.append({
          "Ticker": symbol,
          "Sector": sector,
          "Price": round(current_price, 2),
          "Change_%": round(change_pct, 2),
          "Turnover_Cr": round(avg_turnover_cr, 1),
          "PE": round(pe, 2) if (pe and pe > 0) else np.nan,
          "PB": round(pb, 2) if (pb and pb > 0) else np.nan,
          "ROE_%": round(roe * 100, 1) if roe is not None else np.nan,
          "DE": round(de, 1) if de is not None else np.nan,
          "Rev_Growth_%": (
              round(rev_growth * 100, 1) if rev_growth is not None else np.nan
          ),
          "EPS_Growth_%": (
              round(eps_growth * 100, 1) if eps_growth is not None else np.nan
          ),
          "Mom_6m_1m_%": (
              round(mom_6m_1m, 1) if not np.isnan(mom_6m_1m) else np.nan
          ),
          "Headline_Count": len(clean_titles),
          "Val_Tag": val_tag,
          "Val_Metric": val_metric,
      })
    except Exception as e:
      print(f"Skipping {ticker}: {e}")

  df = pd.DataFrame(records)
  if df.empty:
    return df, hist_data

  finbert_scores = score_headlines_finbert(headlines_map)
  df["FinBERT_Score"] = df["Ticker"].map(finbert_scores)

df["Val_Rank"] = (
        df.groupby("Sector")["Val_Metric"].rank(pct=True, ascending=False) * 100
    ).fillna(0)

    roe_rank = df["ROE_%"].rank(pct=True, ascending=True) * 100
    de_rank = df["DE"].rank(pct=True, ascending=False) * 100
    df["Qual_Rank"] = np.where(
        df["Sector"] == "Financial Services",
        roe_rank,
        (roe_rank.fillna(50) * 0.6 + de_rank.fillna(50) * 0.4),
    )

    rev_rank = df["Rev_Growth_%"].rank(pct=True, ascending=True) * 100
    eps_rank = df["EPS_Growth_%"].rank(pct=True, ascending=True) * 100
    df["Growth_Rank"] = rev_rank.fillna(50) * 0.5 + eps_rank.fillna(50) * 0.5

    mom_rank = df["Mom_6m_1m_%"].rank(pct=True, ascending=True).fillna(0.5) * 100
    sent_rank = df["FinBERT_Score"].rank(pct=True, ascending=True).fillna(0.5) * 100
    df["Mom_Sent_Rank"] = mom_rank * 0.6 + sent_rank * 0.4

    df["Composite_Score"] = (
        df["Val_Rank"] * 0.25
        + df["Qual_Rank"].fillna(50) * 0.30
        + df["Growth_Rank"].fillna(50) * 0.20
        + df["Mom_Sent_Rank"] * 0.25
    ).round(1)

    df.loc[df["Val_Tag"] != "Valid", "Composite_Score"] = (
        df["Composite_Score"] * 0.5
    ).round(1)

    for col in ["Val_Rank", "Qual_Rank", "Growth_Rank", "Mom_Sent_Rank"]:
      df[col] = df[col].fillna(50).round(1)

    quintile_cutoff = df["Composite_Score"].quantile(0.80)

    def assign_flag(row):
      if row["Val_Tag"] != "Valid":
        return row["Val_Tag"]
      elif row["Composite_Score"] >= quintile_cutoff:
        return "Passes Multi-Factor Screen"
      elif row["Composite_Score"] >= 50:
        return "Neutral / Watchlist"
      else:
        return "Fails Screen / Lagging"

    df["Screen_Flag"] = df.apply(assign_flag, axis=1)
    df = df.sort_values(by="Composite_Score", ascending=False).reset_index(
        drop=True
    )
    return df, hist_data

def run_forward_backtest(history_df, hist_data):
  """Evaluates historical snapshots against Nifty 50 (^NSEI) net of 0.25% NSE delivery costs."""
  summary = {
      "status": "accumulating",
      "first_date": "N/A",
      "unique_days": 0,
      "horizon_days": 0,
      "top_net_ret": 0.0,
      "nifty_ret": 0.0,
      "net_alpha": 0.0,
      "hit_rate": 0.0,
      "spearman_ic": 0.0,
      "spread": 0.0,
  }
  if history_df.empty or "Composite_Score" not in history_df.columns:
    return summary

  valid_hist = history_df.dropna(subset=["Composite_Score", "Price"]).copy()
  unique_dates = sorted(valid_hist["Run_Date"].unique())
  if not unique_dates:
    return summary

  summary["first_date"] = unique_dates[0]
  summary["unique_days"] = len(unique_dates)

  # Pick the earliest available snapshot date that is at least 5 calendar days old
  latest_date = pd.to_datetime(unique_dates[-1])
  eligible_dates = [
      d for d in unique_dates if (latest_date - pd.to_datetime(d)).days >= 5
  ]
  if not eligible_dates:
    return summary

  eval_date = eligible_dates[0]
  summary["horizon_days"] = (latest_date - pd.to_datetime(eval_date)).days
  cohort = valid_hist[valid_hist["Run_Date"] == eval_date].copy()

  # Map current prices from hist_data
  curr_prices = {}
  for t in cohort["Ticker"].unique():
    ns_t = f"{t}.NS"
    if ns_t in hist_data.columns.get_level_values(0):
      s = hist_data[ns_t]["Close"].dropna()
      if not s.empty:
        curr_prices[t] = float(s.iloc[-1])

  cohort["Curr_Price"] = cohort["Ticker"].map(curr_prices)
  cohort = cohort.dropna(subset=["Curr_Price"])
  if len(cohort) < 10:
    return summary

  # Gross forward return (%)
  cohort["Fwd_Ret_%"] = (
      (cohort["Curr_Price"] - cohort["Price"]) / cohort["Price"]
  ) * 100

  # Nifty 50 (^NSEI) benchmark return over the exact same window
  nifty_series = hist_data["^NSEI"]["Close"].dropna()
  nifty_series.index = nifty_series.index.tz_localize(None)
  start_dt = pd.to_datetime(eval_date)
  nifty_sub = nifty_series[nifty_series.index >= start_dt]
  if len(nifty_sub) < 2:
    return summary
  nifty_ret = (
      (float(nifty_sub.iloc[-1]) - float(nifty_sub.iloc[0]))
      / float(nifty_sub.iloc[0])
  ) * 100

  # Deduct 0.25% round-trip NSE delivery transaction cost (STT, stamp duty, brokerage, slippage)
  tx_cost = 0.25
  top_q = cohort[cohort["Screen_Flag"] == "Passes Multi-Factor Screen"]
  bot_q = cohort[cohort["Screen_Flag"] == "Fails Screen / Lagging"]

  if top_q.empty:
    return summary

  top_net_ret = float(top_q["Fwd_Ret_%"].mean()) - tx_cost
  bot_net_ret = float(bot_q["Fwd_Ret_%"].mean()) - tx_cost if not bot_q.empty else 0.0
  hit_rate = float((top_q["Fwd_Ret_%"] - tx_cost > nifty_ret).mean()) * 100
  ic = float(
      cohort["Composite_Score"].corr(cohort["Fwd_Ret_%"], method="spearman")
  )

  summary.update({
      "status": "active",
      "eval_date": eval_date,
      "top_net_ret": round(top_net_ret, 2),
      "nifty_ret": round(nifty_ret, 2),
      "net_alpha": round(top_net_ret - nifty_ret, 2),
      "hit_rate": round(hit_rate, 1),
      "spearman_ic": round(ic, 3) if not np.isnan(ic) else 0.0,
      "spread": round(top_net_ret - bot_net_ret, 2),
  })
  return summary


def generate_html(df, bt):
  ist = timezone(timedelta(hours=5, minutes=30))
  timestamp = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")

  total_screened = len(df)
  top_quintile_count = len(df[df["Screen_Flag"] == "Passes Multi-Factor Screen"])
  nm_count = len(df[df["Val_Tag"] != "Valid"])
  finbert_covered = int(df["FinBERT_Score"].notna().sum())

  if bt["status"] == "active":
    alpha_color = "text-emerald-400" if bt["net_alpha"] >= 0 else "text-rose-400"
    verdict = (
        "OUTPERFORMING NIFTY 50"
        if bt["net_alpha"] >= 0
        else "TRAILING NIFTY 50"
    )
    bt_banner = f"""
        <div class="bg-slate-900 border border-emerald-500/30 rounded-xl p-5 mb-6">
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-2 mb-4">
                <div>
                    <span class="text-xs font-bold uppercase tracking-wider text-emerald-400">Live Point-in-Time Backtest vs. Nifty 50 (^NSEI)</span>
                    <h2 class="text-lg font-bold text-white">Cohort Recorded: {bt['eval_date']} ({bt['horizon_days']}-Day Forward Window • Net of 0.25% NSE Tx Costs)</h2>
                </div>
                <span class="px-3 py-1 rounded-full text-xs font-bold bg-slate-800 {alpha_color} border border-slate-700">{verdict}</span>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
                <div class="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <div class="text-xs text-slate-400">Top Quintile (Net)</div>
                    <div class="text-lg font-bold text-white">{bt['top_net_ret']:+.2f}%</div>
                </div>
                <div class="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <div class="text-xs text-slate-400">Nifty 50 Benchmark</div>
                    <div class="text-lg font-bold text-slate-300">{bt['nifty_ret']:+.2f}%</div>
                </div>
                <div class="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <div class="text-xs text-slate-400">Net Alpha vs. Nifty</div>
                    <div class="text-lg font-bold {alpha_color}">{bt['net_alpha']:+.2f}%</div>
                </div>
                <div class="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <div class="text-xs text-slate-400">Hit Rate / Spearman IC</div>
                    <div class="text-lg font-bold text-sky-400">{bt['hit_rate']}% <span class="text-xs text-slate-400">/ IC {bt['spearman_ic']}</span></div>
                </div>
                <div class="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <div class="text-xs text-slate-400">Top vs. Bottom Spread</div>
                    <div class="text-lg font-bold text-emerald-300">{bt['spread']:+.2f}%</div>
                </div>
            </div>
        </div>
        """
  else:
    bt_banner = f"""
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 mb-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
            <div>
                <span class="text-xs font-semibold uppercase tracking-wider text-sky-400">Point-in-Time Forward Backtest Engine Active</span>
                <p class="text-sm text-slate-300 mt-0.5">Recording daily snapshots since <strong>{bt['first_date']}</strong> ({bt['unique_days']} trading day logged). Nifty 50 (^NSEI) net alpha, hit rate, and Spearman IC automatically unlock once the first cohort reaches <strong>5+ days</strong>.</p>
            </div>
            <span class="text-xs font-mono bg-slate-800 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700">Zero Look-Ahead Bias</span>
        </div>
        """

  records_json = df.fillna("N/A").to_json(orient="records")

  html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSE Multi-Factor & FinBERT Quantitative Screener</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-4 md:p-8 font-sans antialiased">
    <div class="max-w-7xl mx-auto">
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 border-b border-slate-800 pb-5 gap-4">
            <div>
                <span class="text-xs font-semibold uppercase tracking-wider text-emerald-400">Institutional Quantitative & NLP Research Pipeline</span>
                <h1 class="text-2xl md:text-3xl font-bold tracking-tight text-white mt-1">NSE Multi-Factor &amp; FinBERT Screener</h1>
                <p class="text-sm text-slate-400 mt-1">Value (25%) • Quality (30%) • Growth (20%) • Momentum &amp; ProsusAI/FinBERT NLP (25%)</p>
            </div>
            <div class="text-left md:text-right">
                <span class="inline-block bg-slate-900 text-emerald-400 text-xs px-3 py-1.5 rounded-full border border-slate-700 font-mono">Auto-Updated: {timestamp}</span>
            </div>
        </header>

        {bt_banner}

        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div class="text-xs text-slate-400">Liquid Universe Screened</div>
                <div class="text-2xl font-bold text-white mt-1">{total_screened} Stocks</div>
                <div class="text-xs text-emerald-400 mt-1">Min ₹25 Cr Daily Turnover</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div class="text-xs text-slate-400">Top Quintile (Top 20%)</div>
                <div class="text-2xl font-bold text-emerald-400 mt-1">{top_quintile_count} Stocks</div>
                <div class="text-xs text-slate-400 mt-1">Passes Multi-Factor Screen</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div class="text-xs text-slate-400">FinBERT NLP Coverage</div>
                <div class="text-2xl font-bold text-sky-400 mt-1">{finbert_covered} Stocks</div>
                <div class="text-xs text-slate-400 mt-1">≥3 Deduplicated 7D Headlines</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div class="text-xs text-slate-400">Excluded / Outliers</div>
                <div class="text-2xl font-bold text-amber-400 mt-1">{nm_count} Stocks</div>
                <div class="text-xs text-slate-400 mt-1">Tagged NM (Loss / P/E &gt; 100)</div>
            </div>
        </div>
<div class="bg-slate-900 border border-slate-800 rounded-xl p-5 mb-6">
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-2 mb-4">
                <div>
                    <span class="text-xs font-semibold uppercase tracking-wider text-sky-400">What-If Sensitivity Tool (Client-Side Only)</span>
                    <h2 class="text-lg font-bold text-white">Interactive Factor Weight Simulator</h2>
                    <p class="text-xs text-slate-400 mt-0.5">Weights auto-normalize to 100%. Does <strong>not</strong> change Nifty 50 (^NSEI) benchmark returns or overwrite your locked-in point-in-time backtest.</p>
                </div>
                <button id="resetWeightsBtn" type="button" class="px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition">
                    Reset to Default (25 / 30 / 20 / 25)
                </button>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-slate-300 font-medium">Value (P/E &amp; P/B)</span>
                        <span id="wValNorm" class="font-mono text-emerald-400">25.0%</span>
                    </div>
                    <input id="sliderVal" type="range" min="0" max="60" step="5" value="25" class="w-full accent-emerald-400">
                    <div class="text-[11px] text-slate-500 mt-1">Raw slider: <span id="wValRaw">25</span></div>
                </div>

                <div class="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-slate-300 font-medium">Quality (ROE &amp; D/E)</span>
                        <span id="wQualNorm" class="font-mono text-emerald-400">30.0%</span>
                    </div>
                    <input id="sliderQual" type="range" min="0" max="60" step="5" value="30" class="w-full accent-emerald-400">
                    <div class="text-[11px] text-slate-500 mt-1">Raw slider: <span id="wQualRaw">30</span></div>
                </div>

                <div class="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-slate-300 font-medium">Growth (Rev &amp; EPS)</span>
                        <span id="wGrowthNorm" class="font-mono text-emerald-400">20.0%</span>
                    </div>
                    <input id="sliderGrowth" type="range" min="0" max="60" step="5" value="20" class="w-full accent-emerald-400">
                    <div class="text-[11px] text-slate-500 mt-1">Raw slider: <span id="wGrowthRaw">20</span></div>
                </div>

                <div class="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-slate-300 font-medium">Mom + FinBERT (7D)</span>
                        <span id="wMomNorm" class="font-mono text-emerald-400">25.0%</span>
                    </div>
                    <input id="sliderMom" type="range" min="0" max="60" step="5" value="25" class="w-full accent-emerald-400">
                    <div class="text-[11px] text-slate-500 mt-1">Raw slider: <span id="wMomRaw">25</span></div>
                </div>
            </div>
        </div>
    
        <div class="flex flex-wrap gap-3 mb-5">
            <input type="text" id="searchInput" placeholder="Search ticker or sector..." class="bg-slate-900 border border-slate-700 text-sm rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500 w-full md:w-64">
            <select id="flagFilter" class="bg-slate-900 border border-slate-700 text-sm rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500">
                <option value="ALL">All Screen Flags ({total_screened})</option>
                <option value="Passes Multi-Factor Screen">🟢 Top Quintile: Passes Multi-Factor Screen ({top_quintile_count})</option>
                <option value="Neutral / Watchlist">⚪ Neutral / Watchlist</option>
                <option value="Fails Screen / Lagging">🔴 Fails Screen / Lagging</option>
                <option value="NM">⚠️ Not Meaningful (NM Outliers)</option>
            </select>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl shadow-xl overflow-hidden">
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="bg-slate-800/80 border-b border-slate-700 text-xs uppercase tracking-wider text-slate-300 font-semibold">
                            <th class="py-3.5 px-4">Rank &amp; Ticker</th>
                            <th class="py-3.5 px-4">Price</th>
                            <th class="py-3.5 px-4">Composite</th>
                            <th class="py-3.5 px-4">P/E | P/B</th>
                            <th class="py-3.5 px-4">ROE %</th>
                            <th class="py-3.5 px-4">FinBERT (7D)</th>
                            <th class="py-3.5 px-4">Screen Flag</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody" class="divide-y divide-slate-800 text-sm"></tbody>
                </table>
            </div>
        </div>

        <footer class="mt-8 text-xs text-slate-500 border-t border-slate-800 pt-4 space-y-1">
            <p><strong>Methodology:</strong> Cross-sectional percentile ranks computed daily across liquid NSE large/mid-cap equities (20D turnover ≥ ₹25 Cr). Benchmark evaluated against Nifty 50 (^NSEI) net of 0.25% NSE delivery transaction costs using point-in-time logged cohorts.</p>
            <p><strong>Disclaimer:</strong> Educational and quantitative engineering demonstration only. Outputs reflect algorithmic factor filtering, not SEBI-registered investment advice.</p>
        </footer>
    </div>

   <script>
        const stocks = {records_json};
        const sliderIds = ['sliderVal', 'sliderQual', 'sliderGrowth', 'sliderMom'];

        function getNormalizedWeights() {{
            const raw = {{
                v: Number(document.getElementById('sliderVal').value),
                q: Number(document.getElementById('sliderQual').value),
                g: Number(document.getElementById('sliderGrowth').value),
                m: Number(document.getElementById('sliderMom').value)
            }};
            const total = raw.v + raw.q + raw.g + raw.m;
            const norm = total > 0
                ? {{ v: raw.v / total, q: raw.q / total, g: raw.g / total, m: raw.m / total }}
                : {{ v: 0.25, q: 0.30, g: 0.20, m: 0.25 }};

            document.getElementById('wValRaw').textContent = raw.v;
            document.getElementById('wQualRaw').textContent = raw.q;
            document.getElementById('wGrowthRaw').textContent = raw.g;
            document.getElementById('wMomRaw').textContent = raw.m;

            document.getElementById('wValNorm').textContent = (norm.v * 100).toFixed(1) + '%';
            document.getElementById('wQualNorm').textContent = (norm.q * 100).toFixed(1) + '%';
            document.getElementById('wGrowthNorm').textContent = (norm.g * 100).toFixed(1) + '%';
            document.getElementById('wMomNorm').textContent = (norm.m * 100).toFixed(1) + '%';

            return norm;
        }}

        function simulateScores(norm) {{
            const scored = stocks.map(s => {{
                const vRank = s.Val_Rank === 'N/A' ? 0 : Number(s.Val_Rank);
                const qRank = s.Qual_Rank === 'N/A' ? 50 : Number(s.Qual_Rank);
                const gRank = s.Growth_Rank === 'N/A' ? 50 : Number(s.Growth_Rank);
                const mRank = s.Mom_Sent_Rank === 'N/A' ? 50 : Number(s.Mom_Sent_Rank);

                let sim = (norm.v * vRank) + (norm.q * qRank) + (norm.g * gRank) + (norm.m * mRank);
                if (s.Val_Tag !== 'Valid') sim *= 0.5;
                sim = Number(sim.toFixed(1));

                const base = Number(s.Composite_Score);
                const delta = Number((sim - base).toFixed(1));
                return {{ ...s, Sim_Score: sim, Sim_Delta: delta }};
            }});

            const sortedSim = [...scored].map(x => x.Sim_Score).sort((a, b) => a - b);
            const cutoffIdx = Math.floor(sortedSim.length * 0.80);
            const simCutoff = sortedSim[cutoffIdx] ?? 70;

            return scored.map(x => ({{
                ...x,
                In_Sim_Top20: x.Val_Tag === 'Valid' && x.Sim_Score >= simCutoff
            }}));
        }}

        function renderTable() {{
            const query = document.getElementById('searchInput').value.toLowerCase();
            const flagFilter = document.getElementById('flagFilter').value;
            const normWeights = getNormalizedWeights();
            const simStocks = simulateScores(normWeights);

            const filtered = simStocks.filter(s => {{
                const matchText = s.Ticker.toLowerCase().includes(query) || s.Sector.toLowerCase().includes(query);
                const matchFlag = flagFilter === 'ALL' || (flagFilter === 'NM' ? s.Screen_Flag.startsWith('NM') : s.Screen_Flag === flagFilter);
                return matchText && matchFlag;
            }});

            document.getElementById('tableBody').innerHTML = filtered.map((s, idx) => {{
                let badgeClass = 'bg-slate-800 text-slate-300';
                if (s.Screen_Flag === 'Passes Multi-Factor Screen') badgeClass = 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30';
                else if (s.Screen_Flag.startsWith('NM')) badgeClass = 'bg-amber-500/15 text-amber-400 border border-amber-500/30';
                else if (s.Screen_Flag === 'Fails Screen / Lagging') badgeClass = 'bg-rose-500/10 text-rose-400';

                const chgColor = s['Change_%'] >= 0 ? 'text-emerald-400' : 'text-rose-400';
                const deltaColor = s.Sim_Delta > 0 ? 'text-emerald-400' : (s.Sim_Delta < 0 ? 'text-rose-400' : 'text-slate-400');
                const simBadge = s.In_Sim_Top20
                    ? '<span class="ml-1.5 px-1.5 py-0.5 text-[10px] font-semibold rounded bg-sky-500/15 text-sky-300 border border-sky-500/30">Sim Top 20%</span>'
                    : '';

                let sentHtml = '<span class="text-slate-500 text-xs">Insufficient (&lt;3)</span>';
                if (s.FinBERT_Score !== 'N/A') {{
                    const sc = Number(s.FinBERT_Score);
                    const sColor = sc > 0.1 ? 'text-emerald-400' : (sc < -0.1 ? 'text-rose-400' : 'text-slate-300');
                    sentHtml = `<span class="font-mono font-semibold ${{sColor}}">${{sc >= 0 ? '+' : ''}}${{sc.toFixed(2)}}</span> <span class="text-xs text-slate-500">(${{s.Headline_Count}} news)</span>`;
                }}

                return `
                    <tr class="hover:bg-slate-800/40 transition">
                        <td class="py-3 px-4">
                            <div class="font-bold text-white"><span class="text-slate-500 font-mono mr-1.5">#${{idx + 1}}</span>${{s.Ticker}}${{simBadge}}</div>
                            <div class="text-xs text-slate-400">${{s.Sector}}</div>
                        </td>
                        <td class="py-3 px-4">
                            <div class="font-semibold text-white">₹${{Number(s.Price).toLocaleString('en-IN')}}</div>
                            <div class="text-xs ${{chgColor}}">${{s['Change_%'] >= 0 ? '+' : ''}}${{s['Change_%']}}%</div>
                        </td>
                        <td class="py-3 px-4 font-mono">
                            <div class="font-bold text-emerald-300">${{s.Composite_Score}} <span class="text-xs text-slate-500 font-normal">/100</span></div>
                            <div class="text-xs text-sky-300">Sim: ${{s.Sim_Score}} <span class="${{deltaColor}}">(${{s.Sim_Delta >= 0 ? '+' : ''}}${{s.Sim_Delta}})</span></div>
                        </td>
                        <td class="py-3 px-4 text-slate-300 text-xs font-mono">PE: ${{s.PE}}<br>PB: ${{s.PB}}</td>
                        <td class="py-3 px-4 text-slate-300">${{s['ROE_%'] !== 'N/A' ? s['ROE_%'] + '%' : 'N/A'}}</td>
                        <td class="py-3 px-4">${{sentHtml}}</td>
                        <td class="py-3 px-4"><span class="px-2.5 py-1 rounded-full text-xs font-medium ${{badgeClass}}">${{s.Screen_Flag}}</span></td>
                    </tr>
                `;
            }}).join('');
        }}

        renderTable();
        sliderIds.forEach(id => document.getElementById(id).addEventListener('input', renderTable));
        document.getElementById('resetWeightsBtn').addEventListener('click', () => {{
            document.getElementById('sliderVal').value = 25;
            document.getElementById('sliderQual').value = 30;
            document.getElementById('sliderGrowth').value = 20;
            document.getElementById('sliderMom').value = 25;
            renderTable();
        }});
        document.getElementById('searchInput').addEventListener('input', renderTable);
        document.getElementById('flagFilter').addEventListener('change', renderTable);
        </script>
</body>
</html>
"""
  return html_content


if __name__ == "__main__":
  df, hist_data = fetch_universe_data()

  hf_token = os.environ.get("HF_TOKEN")
  repo_id = "Anujk04/indian-stock-screener"
  history_file = "screener_history.csv"
  existing_df = pd.DataFrame()

  if hf_token:
    api = HfApi()
    try:
      downloaded_path = hf_hub_download(
          repo_id=repo_id,
          filename=history_file,
          repo_type="space",
          token=hf_token,
      )
      existing_df = pd.read_csv(downloaded_path)
    except Exception:
      existing_df = pd.DataFrame()

    ist = timezone(timedelta(hours=5, minutes=30))
    df["Run_Date"] = datetime.now(ist).strftime("%Y-%m-%d")
    df["Timestamp"] = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")

    combined_df = pd.concat([existing_df, df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(
        subset=["Run_Date", "Ticker"], keep="last"
    )
    combined_df.to_csv(history_file, index=False)
  else:
    combined_df = df

  bt_summary = run_forward_backtest(combined_df, hist_data)
  html = generate_html(df, bt_summary)

  with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

  if hf_token:
    api.upload_file(
        path_or_fileobj="index.html",
        path_in_repo="index.html",
        repo_id=repo_id,
        repo_type="space",
        token=hf_token,
    )
    api.upload_file(
        path_or_fileobj=history_file,
        path_in_repo=history_file,
        repo_id=repo_id,
        repo_type="space",
        token=hf_token,
    )
    print("Successfully updated FinBERT dashboard & live Nifty 50 backtester!")
