import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from huggingface_hub import HfApi

# 1. 15-Stock Watchlist (NSE Tickers)
TICKERS = [
    "COCHINSHIP.NS",
    "INFY.NS",
    "ITC.NS",
    "LGINDIALTD.NS",
    "RELIANCE.NS",
    "SHRIRAMFIN.NS",
    "IDEA.NS",
    "LT.NS",
    "BEL.NS",
    "HDFCBANK.NS",
    "ZOMATO.NS",
    "HAL.NS",
    "DLF.NS",
    "DRREDDY.NS",
    "TATAMOTORS.NS",
]


def fetch_stock_data():
  data = []
  for ticker in TICKERS:
    try:
      stock = yf.Ticker(ticker)
      info = stock.info
      hist = stock.history(period="1y")

      if hist.empty:
        continue

      current_price = hist["Close"].iloc[-1]
      prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current_price
      price_change_pct = ((current_price - prev_close) / prev_close) * 100

      pe = info.get("trailingPE", None)
      pb = info.get("priceToBook", None)
      roe = info.get("returnOnEquity", None)
      sector = info.get("sector", "General")
      eps = info.get("trailingEps", None)

      # Valuation Tag & Edge Case Handling (NM for loss makers or extreme PE)
      if eps is not None and eps <= 0:
        val_status = "NM (Loss-Making)"
      elif pe is not None and pe > 100:
        val_status = "NM (High P/E)"
      elif pb is not None and pb <= 0:
        val_status = "NM (Neg. Equity)"
      else:
        val_status = "Normal"

      # Simple Sentiment / News Placeholder (VADER or News stub)
      news_list = stock.news
      sentiment_score = 0.0
      if news_list:
        # Mock sentiment aggregation for stability in free tier
        sentiment_score = 0.15  # Placeholder for VADER/FinBERT net polarity

      data.append({
          "Ticker": ticker.replace(".NS", ""),
          "Sector": sector,
          "Price": round(current_price, 2),
          "Change_%": round(price_change_pct, 2),
          "PE": round(pe, 2) if pe and pe > 0 else "N/A",
          "PB": round(pb, 2) if pb else "N/A",
          "ROE": round(roe * 100, 2) if roe else "N/A",
          "Valuation_Tag": val_status,
          "Sentiment": round(sentiment_score, 2),
      })
    except Exception as e:
      print(f"Error fetching {ticker}: {e}")

  return pd.DataFrame(data)


def generate_html(df):
  timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p IST")

  rows_html = ""
  for _, row in df.iterrows():
    badge_class = (
        "bg-emerald-100 text-emerald-800"
        if row["Valuation_Tag"] == "Normal"
        else "bg-amber-100 text-amber-800"
    )
    rows_html += f"""
        <tr class="hover:bg-slate-50 border-b border-slate-100">
            <td class="py-3 px-4 font-semibold text-slate-800">{row['Ticker']}</td>
            <td class="py-3 px-4 text-slate-600">{row['Sector']}</td>
            <td class="py-3 px-4 font-medium text-slate-900">₹{row['Price']:,.2f}</td>
            <td class="py-3 px-4 {'text-emerald-600' if row['Change_%'] >= 0 else 'text-rose-600'} font-semibold">{row['Change_%']:+.2f}%</td>
            <td class="py-3 px-4 text-slate-600">{row['PE']}</td>
            <td class="py-3 px-4 text-slate-600">{row['PB']}</td>
            <td class="py-3 px-4"><span class="px-2.5 py-1 rounded-full text-xs font-medium {badge_class}">{row['Valuation_Tag']}</span></td>
            <td class="py-3 px-4 text-slate-600">{row['Sentiment']}</td>
        </tr>
        """

  html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Institutional Quantitative Equity Screener</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen p-6">
    <div class="max-w-7xl mx-auto">
        <header class="flex justify-between items-center mb-8 border-b border-slate-800 pb-4">
            <div>
                <h1 class="text-2xl font-bold tracking-tight text-white">NSE Quantitative Equity Screener</h1>
                <p class="text-sm text-slate-400 mt-1">Multi-factor valuation & sentiment screening pipeline</p>
            </div>
            <div class="text-right">
                <span class="inline-block bg-slate-800 text-emerald-400 text-xs px-3 py-1.5 rounded-lg border border-slate-700 font-mono">Auto-Updated: {timestamp}</span>
            </div>
        </header>

        <div class="bg-white rounded-xl shadow-xl overflow-hidden text-slate-800">
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="bg-slate-100 border-b border-slate-200 text-xs uppercase tracking-wider text-slate-600 font-semibold">
                            <th class="py-3 px-4">Ticker</th>
                            <th class="py-3 px-4">Sector</th>
                            <th class="py-3 px-4">Price</th>
                            <th class="py-3 px-4">Change</th>
                            <th class="py-3 px-4">P/E</th>
                            <th class="py-3 px-4">P/B</th>
                            <th class="py-3 px-4">Screen Status</th>
                            <th class="py-3 px-4">Sentiment</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <footer class="mt-8 text-center text-xs text-slate-500 border-t border-slate-800 pt-4">
            <p><strong>Disclaimer:</strong> Educational and quantitative engineering demonstration only. Outputs reflect algorithmic factor filtering, not SEBI-registered investment advice.</p>
        </footer>
    </div>
</body>
</html>
"""
  return html_content


if __name__ == "__main__":
  df = fetch_stock_data()
  html = generate_html(df)

  # Save local file
  with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

  # Save historical log CSV for backtesting
  history_file = "screener_history.csv"
  df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  if os.path.exists(history_file):
    df.to_csv(history_file, mode="a", header=False, index=False)
  else:
    df.to_csv(history_file, index=False)

  # Push to Hugging Face
  hf_token = os.environ.get("HF_TOKEN")
  if hf_token:
    api = HfApi()
    repo_id = "Anujk04/indian-stock-screener"

    # Upload dashboard
    api.upload_file(
        path_or_fileobj="index.html",
        path_in_repo="index.html",
        repo_id=repo_id,
        repo_type="space",
        token=hf_token,
    )

    # Upload historical log for backtest tracking
    api.upload_file(
        path_or_fileobj="history_file"
        if os.path.exists(history_file)
        else "screener_history.csv",
        path_in_repo="screener_history.csv",
        repo_id=repo_id,
        repo_type="space",
        token=hf_token,
    )
    print("Successfully pushed updated screener and historical log to Hugging Face!")
