from datetime import datetime, timedelta, timezone
import os
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import pandas as pd
import yfinance as yf

# Curated Nifty 100 + Key Watchlist Tickers (ENSURES 100% RELIABLE FETCHING WITHOUT NSE BLOCKS)
UNIVERSE_TICKERS = list(
    set([
        # Original 15 Watchlist (Updated NSE symbols: ETERNAL, TMPV, LGEINDIA)
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
        # Nifty 100 Core Constituents
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


def fetch_universe_data():
  # Bulk download 8 months of daily price/volume data in ONE fast batch call
  print("Downloading bulk price & volume history...")
  hist_data = yf.download(
      UNIVERSE_TICKERS,
      period="8mo",
      group_by="ticker",
      progress=False,
      threads=True,
  )

  records = []
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

      # Liquidity Filter: 20-day Average Daily Turnover >= ₹25 Crore (250,000,000 INR)
      avg_turnover_cr = (
          df_t["Close"].iloc[-20:] * df_t["Volume"].iloc[-20:]
      ).mean() / 1e7
      if avg_turnover_cr < 25.0:
        continue

      # Momentum: 6-month return skipping the most recent 21 trading days (~1 month)
      if len(df_t) >= 130:
        price_6m_ago = float(df_t["Close"].iloc[-130])
        price_1m_ago = float(df_t["Close"].iloc[-21])
        mom_6m_1m = ((price_1m_ago - price_6m_ago) / price_6m_ago) * 100
      else:
        mom_6m_1m = np.nan

      # Fundamental Info
      info = yf.Ticker(ticker).info
      sector = info.get("sector", "General")
      pe = info.get("trailingPE", None)
      pb = info.get("priceToBook", None)
      roe = info.get("returnOnEquity", None)
      de = info.get("debtToEquity", None)
      rev_growth = info.get("revenueGrowth", None)
      eps_growth = info.get("earningsGrowth", None)
      eps = info.get("trailingEps", None)

      # Valuation Meaningfulness Check
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
        # Use P/B for Financials, Trailing P/E for all other sectors
        val_metric = pb if is_financial else pe

      records.append({
          "Ticker": ticker.replace(".NS", ""),
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
          "Val_Tag": val_tag,
          "Val_Metric": val_metric,
      })
    except Exception as e:
      print(f"Skipping {ticker}: {e}")

  df = pd.DataFrame(records)
  if df.empty:
    return df

  # --- CROSS-SECTIONAL MULTI-FACTOR PERCENTILE RANKING (0 to 100) ---
  # 1. Value Score (25%): Lower multiple within sector = higher rank
  df["Val_Rank"] = (
      df.groupby("Sector")["Val_Metric"].rank(pct=True, ascending=False) * 100
  )

  # 2. Quality Score (30%): Higher ROE + Lower Debt/Equity (ignore D/E for Financials)
  roe_rank = df["ROE_%"].rank(pct=True, ascending=True) * 100
  de_rank = df["DE"].rank(pct=True, ascending=False) * 100
  df["Qual_Rank"] = np.where(
      df["Sector"] == "Financial Services",
      roe_rank,
      (roe_rank.fillna(50) * 0.6 + de_rank.fillna(50) * 0.4),
  )

  # 3. Growth Score (20%): Revenue Growth + EPS Growth
  rev_rank = df["Rev_Growth_%"].rank(pct=True, ascending=True) * 100
  eps_rank = df["EPS_Growth_%"].rank(pct=True, ascending=True) * 100
  df["Growth_Rank"] = rev_rank.fillna(50) * 0.5 + eps_rank.fillna(50) * 0.5

  # 4. Momentum Score (25%): 6M-1M Return
  df["Mom_Rank"] = (
      df["Mom_6m_1m_%"].rank(pct=True, ascending=True).fillna(0.5) * 100
  )

  # Weighted Composite Score
  df["Composite_Score"] = (
      df["Val_Rank"].fillna(0) * 0.25
      + df["Qual_Rank"].fillna(50) * 0.30
      + df["Growth_Rank"].fillna(50) * 0.20
      + df["Mom_Rank"].fillna(50) * 0.25
  ).round(1)

  # Penalize NM (Not Meaningful) valuation stocks so they cannot enter Top Quintile
  df.loc[df["Val_Tag"] != "Valid", "Composite_Score"] = (
      df["Composite_Score"] * 0.5
  ).round(1)

  # Top Quintile (Top 20%) Flagging
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

  records_json = df.fillna("N/A").to_json(orient="records")

  html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSE Multi-Factor Quantitative Equity Screener</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-4 md:p-8 font-sans antialiased">
    <div class="max-w-7xl mx-auto">
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 border-b border-slate-800 pb-5 gap-4">
            <div>
                <span class="text-xs font-semibold uppercase tracking-wider text-emerald-400">Institutional Quantitative Research Pipeline</span>
                <h1 class="text-2xl md:text-3xl font-bold tracking-tight text-white mt-1">NSE Multi-Factor Equity Screener</h1>
                <p class="text-sm text-slate-400 mt-1">Cross-Sectional Ranking: Value (25%) • Quality (30%) • Growth (20%) • Momentum (25%)</p>
            </div>
            <div class="text-left md:text-right">
                <span class="inline-block bg-slate-900 text-emerald-400 text-xs px-3 py-1.5 rounded-full border border-slate-700 font-mono">Auto-Updated: {timestamp}</span>
            </div>
        </header>

        <!-- KPI Summary Cards -->
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
                <div class="text-xs text-slate-400">Excluded / Outliers</div>
                <div class="text-2xl font-bold text-amber-400 mt-1">{nm_count} Stocks</div>
                <div class="text-xs text-slate-400 mt-1">Tagged NM (Loss / P/E &gt; 100)</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div class="text-xs text-slate-400">Financials Methodology</div>
                <div class="text-lg font-bold text-white mt-1">P/B + ROE</div>
                <div class="text-xs text-slate-400 mt-1">Sector-Relative Multiples</div>
            </div>
        </div>

        <!-- Filters -->
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

        <!-- Data Table -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl shadow-xl overflow-hidden">
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="bg-slate-800/80 border-b border-slate-700 text-xs uppercase tracking-wider text-slate-300 font-semibold">
                            <th class="py-3.5 px-4">Rank & Ticker</th>
                            <th class="py-3.5 px-4">Price</th>
                            <th class="py-3.5 px-4">Composite</th>
                            <th class="py-3.5 px-4">P/E | P/B</th>
                            <th class="py-3.5 px-4">ROE %</th>
                            <th class="py-3.5 px-4">6M-1M Mom</th>
                            <th class="py-3.5 px-4">Screen Flag</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody" class="divide-y divide-slate-800 text-sm"></tbody>
                </table>
            </div>
        </div>

        <footer class="mt-8 text-xs text-slate-500 border-t border-slate-800 pt-4 space-y-1">
            <p><strong>Methodology:</strong> Cross-sectional percentile ranks computed daily across liquid NSE large/mid-cap equities (20D turnover ≥ ₹25 Cr). Financial Services evaluated on P/B &amp; ROE; non-financials on sector-relative P/E, ROE, Debt/Equity, Revenue/EPS growth, and 6M-1M momentum. Loss-makers, negative equity, and P/E &gt; 100 are tagged Not Meaningful (NM).</p>
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
                const momVal = s['Mom_6m_1m_%'] !== 'N/A' ? s['Mom_6m_1m_%'] + '%' : 'N/A';

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
                        <td class="py-3 px-4 text-slate-300">${{momVal}}</td>
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
    # Download existing history from Hugging Face so logs persist across GitHub Actions runs
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
    # Deduplicate same-day manual test runs so you keep 1 clean snapshot per calendar day
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
    print(f"Successfully updated dashboard with {len(df)} liquid stocks!")
