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
  """Fetches Google News RSS headlines from the last 7 days, strips spam & duplicates."""
  query = urllib.parse.quote(f'"{company_name}" OR "{symbol}" stock India')
  url = (
      f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
  )
  feed = feedparser.parse(url)

  cutoff = datetime.now(timezone.utc) - timedelta(days=7)
  clean_titles = []

  for entry in feed.entries[:25]:
    # 1. Freshness check (last 7 days)
    if hasattr(entry, "published"):
      try:
        pub_dt = parsedate_to_datetime(entry.published)
        if pub_dt < cutoff:
          continue
      except Exception:
        pass

    # 2. Spam aggregator filter
    source_href = (
        entry.get("source", {}).get("href", "").lower()
        if isinstance(entry.get("source"), dict)
        else ""
    )
    if any(spam in source_href for spam in SPAM_DOMAINS):
      continue

    # Strip trailing publisher name (" - Economic Times")
    title = entry.title.rsplit(" - ", 1)[0].strip()
    if len(title) < 15:
      continue

    # 3. Fuzzy deduplication (ratio > 0.75)
    is_dup = any(
        SequenceMatcher(None, title.lower(), existing.lower()).ratio() > 0.75
        for existing in clean_titles
    )
    if not is_dup:
      clean_titles.append(title)

  return clean_titles[:10]


def score_headlines_finbert(ticker_headlines_map):
  """Runs batched ProsusAI/finbert inference across all valid headlines."""
  print("Loading ProsusAI/finbert model...")
  model_name = "ProsusAI/finbert"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)
  model.eval()

  # Flatten all headlines for fast batched inference
  flat_texts = []
  ticker_slices = {}
  idx = 0
  for ticker, titles in ticker_headlines_map.items():
    if len(titles) >= 3:  # Minimum 3 unique headlines required
      ticker_slices[ticker] = (idx, idx + len(titles))
      flat_texts.extend(titles)
      idx += len(titles)

  scores = {}
  if not flat_texts:
    return scores

  batch_size = 32
  net_polarities = []
  with torch.no_grad():
    for i in range(0, len(flat_texts), batch_size):
      batch = flat_texts[i : i + batch_size]
      inputs = tokenizer(
          batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
      )
      outputs = model(**inputs)
      probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
      # ProsusAI/finbert label order: 0 = positive, 1 = negative, 2 = neutral
      for p in probs:
        net_polarities.append(float(p[0] - p[1]))

  for ticker, (start, end) in ticker_slices.items():
    scores[ticker] = round(float(np.mean(net_polarities[start:end])), 3)

  return scores


def fetch_universe_data():
  print("Downloading bulk price & volume history...")
  hist_data = yf.download(
      UNIVERSE_TICKERS,
      period="8mo",
      group_by="ticker",
      progress=False,
      threads=True,
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
    return df

  # Run FinBERT on all cleaned headlines
  finbert_scores = score_headlines_finbert(headlines_map)
  df["FinBERT_Score"] = df["Ticker"].map(finbert_scores)  # NaN if < 3 headlines

  # --- CROSS-SECTIONAL MULTI-FACTOR PERCENTILE RANKING (0 to 100) ---
  df["Val_Rank"] = (
      df.groupby("Sector")["Val_Metric"].rank(pct=True, ascending=False) * 100
  )

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

  # Momentum + FinBERT Sentiment (25% total: 15% Price Momentum + 10% FinBERT NLP)
  mom_rank = df["Mom_6m_1m_%"].rank(pct=True, ascending=True).fillna(0.5) * 100
  sent_rank = df["FinBERT_Score"].rank(pct=True, ascending=True).fillna(0.5) * 100
  df["Mom_Sent_Rank"] = mom_rank * 0.6 + sent_rank * 0.4

  df["Composite_Score"] = (
      df["Val_Rank"].fillna(0) * 0.25
      + df["Qual_Rank"].fillna(50) * 0.30
      + df["Growth_Rank"].fillna(50) * 0.20
      + df["Mom_Sent_Rank"] * 0.25
  ).round(1)

  df.loc[df["Val_Tag"] != "Valid", "Composite_Score"] = (
      df["Composite_Score"] * 0.5
  ).round(1)

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
  return df


def generate_html(df):
  ist = timezone(timedelta(hours=5, minutes=30))
  timestamp = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")

  total_screened = len(df)
  top_quintile_count = len(df[df["Screen_Flag"] == "Passes Multi-Factor Screen"])
  nm_count = len(df[df["Val_Tag"] != "Valid"])
  finbert_covered = int(df["FinBERT_Score"].notna().sum())

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
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 border-b border-slate-800 pb-5 gap-4">
            <div>
                <span class="text-xs font-semibold uppercase tracking-wider text-emerald-400">Institutional Quantitative & NLP Research Pipeline</span>
                <h1 class="text-2xl md:text-3xl font-bold tracking-tight text-white mt-1">NSE Multi-Factor &amp; FinBERT Screener</h1>
                <p class="text-sm text-slate-400 mt-1">Value (25%) • Quality (30%) • Growth (20%) • Momentum &amp; ProsusAI/FinBERT NLP (25%)</p>
            </div>
            <div class="text-left md:text-right">
                <span class="inline-block bg-slate-900 text-emerald-400 text-xs px-3 py-1.5 rounded-full border border-slate-700 font-mono">Auto-Updated: {timestamp}</span>
            </div>
        </header>

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
            <p><strong>Methodology:</strong> Cross-sectional percentile ranks computed daily across liquid NSE large/mid-cap equities (20D turnover ≥ ₹25 Cr). Financial Services evaluated on P/B &amp; ROE; non-financials on sector-relative P/E, ROE, Debt/Equity, Revenue/EPS growth, 6M-1M momentum, and ProsusAI/FinBERT net sentiment across deduplicated 7-day headlines (minimum 3 headlines required).</p>
            <p><strong>Disclaimer:</strong> Educational and quantitative engineering demonstration only. Outputs reflect algorithmic factor filtering, not SEBI-registered investment advice.</p>
        </footer>
    </div>

    <script>
        const stocks = {records_json};

        function renderTable() {{
            const query = document.getElementById('searchInput').value.toLowerCase();
            const flagFilter = document.getElementById('flagFilter').value;

            const filtered = stocks.filter(s => {{
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
                let sentHtml = '<span class="text-slate-500 text-xs">Insufficient (&lt;3)</span>';
                if (s.FinBERT_Score !== 'N/A') {{
                    const sc = Number(s.FinBERT_Score);
                    const sColor = sc > 0.1 ? 'text-emerald-400' : (sc < -0.1 ? 'text-rose-400' : 'text-slate-300');
                    sentHtml = `<span class="font-mono font-semibold ${{sColor}}">${{sc >= 0 ? '+' : ''}}${{sc.toFixed(2)}}</span> <span class="text-xs text-slate-500">(${{s.Headline_Count}} news)</span>`;
                }}

                return `
                    <tr class="hover:bg-slate-800/40 transition">
                        <td class="py-3 px-4">
                            <div class="font-bold text-white"><span class="text-slate-500 font-mono mr-1.5">#${{idx + 1}}</span>${{s.Ticker}}</div>
                            <div class="text-xs text-slate-400">${{s.Sector}}</div>
                        </td>
                        <td class="py-3 px-4">
                            <div class="font-semibold text-white">₹${{Number(s.Price).toLocaleString('en-IN')}}</div>
                            <div class="text-xs ${{chgColor}}">${{s['Change_%'] >= 0 ? '+' : ''}}${{s['Change_%']}}%</div>
                        </td>
                        <td class="py-3 px-4 font-mono font-bold text-emerald-300">${{s.Composite_Score}} <span class="text-xs text-slate-500 font-normal">/100</span></td>
                        <td class="py-3 px-4 text-slate-300 text-xs font-mono">PE: ${{s.PE}}<br>PB: ${{s.PB}}</td>
                        <td class="py-3 px-4 text-slate-300">${{s['ROE_%'] !== 'N/A' ? s['ROE_%'] + '%' : 'N/A'}}</td>
                        <td class="py-3 px-4">${{sentHtml}}</td>
                        <td class="py-3 px-4"><span class="px-2.5 py-1 rounded-full text-xs font-medium ${{badgeClass}}">${{s.Screen_Flag}}</span></td>
                    </tr>
                `;
            }}).join('');
        }}

        renderTable();
        document.getElementById('searchInput').addEventListener('input', renderTable);
        document.getElementById('flagFilter').addEventListener('change', renderTable);
    </script>
</body>
</html>
"""
  return html_content


if __name__ == "__main__":
  df = fetch_universe_data()
  html = generate_html(df)

  with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

  hf_token = os.environ.get("HF_TOKEN")
  repo_id = "Anujk04/indian-stock-screener"
  history_file = "screener_history.csv"

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
    print(f"Successfully updated FinBERT dashboard with {len(df)} stocks!")
