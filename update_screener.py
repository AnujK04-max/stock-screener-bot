import yfinance as yf
import urllib.request
import xml.etree.ElementTree as ET
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import json
from datetime import datetime, timezone, timedelta

nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

WATCHLIST = {
    "Cochin Shipyard": {"ticker": "COCHINSHIP.NS", "query": "Cochin Shipyard stock", "sector": "Defense & Shipbuilding"},
    "Infosys": {"ticker": "INFY.NS", "query": "Infosys stock NSE", "sector": "IT Services"},
    "ITC": {"ticker": "ITC.NS", "query": "ITC limited stock", "sector": "FMCG & Conglomerate"},
    "LG Electronics India": {"ticker": "LGEINDIA.NS", "query": "LG Electronics India stock", "sector": "Consumer Durables"},
    "Reliance Industries": {"ticker": "RELIANCE.NS", "query": "Reliance Industries stock", "sector": "Energy & Retail"},
    "Shriram Finance": {"ticker": "SHRIRAMFIN.NS", "query": "Shriram Finance stock", "sector": "NBFC / Finance"},
    "Vodafone Idea": {"ticker": "IDEA.NS", "query": "Vodafone Idea stock", "sector": "Telecom"},
    "Larsen & Toubro": {"ticker": "LT.NS", "query": "LT stock India", "sector": "Infrastructure"},
    "Bharat Electronics": {"ticker": "BEL.NS", "query": "BEL stock India", "sector": "Defense Electronics"},
    "HDFC Bank": {"ticker": "HDFCBANK.NS", "query": "HDFC Bank stock", "sector": "Banking"},
    "Eternal (Zomato)": {"ticker": "ETERNAL.NS", "query": "Eternal Zomato stock", "sector": "Consumer Tech"},
    "Hindustan Aeronautics": {"ticker": "HAL.NS", "query": "HAL stock India", "sector": "Aerospace & Defense"},
    "DLF": {"ticker": "DLF.NS", "query": "DLF stock India", "sector": "Real Estate"},
    "Dr. Reddy's Labs": {"ticker": "DRREDDY.NS", "query": "Dr Reddys Lab stock", "sector": "Pharmaceuticals"},
    "Tata Motors PV": {"ticker": "TMPV.NS", "query": "Tata Motors TMPV stock", "sector": "Automobiles"}
}

data_records = []

for name, info_dict in WATCHLIST.items():
    ticker = info_dict["ticker"]
    query = info_dict["query"]
    sector = info_dict["sector"]
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y").dropna(subset=['Close'])
        if hist.empty:
            stock = yf.Ticker(ticker.replace(".NS", ".BO"))
            hist = stock.history(period="1y").dropna(subset=['Close'])
            
        info = stock.info
        if not hist.empty:
            price = round(float(hist['Close'].iloc[-1]), 2)
            high_52w = round(float(hist['High'].dropna().max()), 2)
        else:
            price = round(float(info.get('currentPrice') or info.get('regularMarketPrice') or 0), 2)
            high_52w = round(float(info.get('fiftyTwoWeekHigh') or price), 2)

        pct_high = round(((price - high_52w) / high_52w) * 100, 1) if high_52w > 0 else 0.0
        pe = info.get('trailingPE', None)
        f_pe = info.get('forwardPE', None)
        pb = info.get('priceToBook', None)
        
        if pe is None or pe <= 0:
            val_status = "N/A / Loss"
        elif f_pe and pe and f_pe < pe * 0.85:
            val_status = "Undervalued (Growth)"
        elif pe < 22:
            val_status = "Value Zone"
        elif pe > 55:
            val_status = "High Growth"
        else:
            val_status = "Fairly Valued"
    except:
        price, high_52w, pct_high, pe, pb, val_status = 0, 0, 0, None, None, "Error"

    headlines_list = []
    try:
        formatted_query = query.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={formatted_query}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        root = ET.fromstring(response.read())
        
        scores = []
        for item in root.findall('.//item')[:4]:
            title_el = item.find('title')
            link_el = item.find('link')
            if title_el is not None and title_el.text:
                score = round(sid.polarity_scores(title_el.text)['compound'], 2)
                scores.append(score)
                headlines_list.append({
                    "title": title_el.text,
                    "link": link_el.text if link_el is not None else "#",
                    "score": score
                })
                
        if scores:
            avg_score = round(sum(scores) / len(scores), 2)
            news_mood = "Bullish" if avg_score >= 0.15 else ("Bearish" if avg_score <= -0.15 else "Neutral")
        else:
            avg_score, news_mood = 0.0, "Neutral"
    except:
        avg_score, news_mood = 0.0, "Neutral"

    if "Value" in val_status or "Undervalued" in val_status:
        action = "Strong Buy / Accumulate" if news_mood == "Bullish" else "Deep Value Watch"
    elif news_mood == "Bearish" and val_status == "High Growth":
        action = "Caution / Trim"
    elif news_mood == "Bullish":
        action = "Momentum Watch"
    else:
        action = "Hold / Neutral"

    data_records.append({
        "company": name,
        "ticker": ticker.replace(".NS", ""),
        "sector": sector,
        "price": price,
        "high_52w": high_52w,
        "pct_high": pct_high,
        "pe": round(pe, 1) if pe else "N/A",
        "pb": round(pb, 2) if pb else "N/A",
        "val_status": val_status,
        "sentiment_score": avg_score,
        "news_mood": news_mood,
        "action": action,
        "headlines": headlines_list
    })

ist = timezone(timedelta(hours=5, minutes=30))
last_updated = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")
json_data = json.dumps(data_records)

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Indian Equity Quantitative & Sentiment Screener</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 font-sans antialiased">
  <div class="max-w-7xl mx-auto px-4 py-8">
    <div class="flex flex-col md:flex-row md:items-center md:justify-between border-b border-slate-800 pb-6 mb-8">
      <div>
        <span class="text-xs font-semibold uppercase tracking-wider text-emerald-400">Quantitative Equity Research Project</span>
        <h1 class="text-3xl font-bold text-white mt-1">Indian Equity Valuation & Sentiment Screener</h1>
        <p class="text-slate-400 text-sm mt-1">Combining Relative Valuation Multiples (P/E, P/B) with NLP News Sentiment across 15 NSE Stocks</p>
      </div>
      <div class="mt-4 md:mt-0 text-left md:text-right">
        <span class="inline-block bg-slate-900 border border-slate-700 text-slate-300 text-xs px-3 py-1.5 rounded-full">
          Auto-Updated: {last_updated}
        </span>
      </div>
    </div>
    <div id="kpi-cards" class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"></div>
    <div class="flex flex-wrap gap-3 mb-6">
      <input type="text" id="searchInput" placeholder="Search company or sector..." class="bg-slate-900 border border-slate-700 text-sm rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500 w-full md:w-64" />
      <select id="moodFilter" class="bg-slate-900 border border-slate-700 text-sm rounded-lg px-4 py-2 text-white focus:outline-none focus:border-emerald-500">
        <option value="ALL">All News Sentiments</option>
        <option value="Bullish">🟢 Bullish Sentiment</option>
        <option value="Neutral">⚪ Neutral Sentiment</option>
        <option value="Bearish">🔴 Bearish Sentiment</option>
      </select>
    </div>
    <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl mb-10">
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse">
          <thead>
            <tr class="bg-slate-800/70 text-slate-300 text-xs uppercase tracking-wider border-b border-slate-700">
              <th class="py-3.5 px-4">Company</th>
              <th class="py-3.5 px-4">Price (₹)</th>
              <th class="py-3.5 px-4">52W High</th>
              <th class="py-3.5 px-4">Trailing P/E</th>
              <th class="py-3.5 px-4">Valuation Tag</th>
              <th class="py-3.5 px-4">News Mood</th>
              <th class="py-3.5 px-4">Composite Signal</th>
            </tr>
          </thead>
          <tbody id="tableBody" class="divide-y divide-slate-800 text-sm"></tbody>
        </table>
      </div>
    </div>
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-6">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div>
          <h2 class="text-xl font-bold text-white">🔍 Stock Deep-Dive & NLP Headline Inspector</h2>
          <p class="text-slate-400 text-sm">Select any stock below (or click a row above) to inspect VADER-scored headlines</p>
        </div>
        <select id="stockSelect" class="bg-slate-800 border border-slate-700 text-white text-sm rounded-lg px-4 py-2.5 focus:outline-none focus:border-emerald-500"></select>
      </div>
      <div id="headlineContainer" class="space-y-3"></div>
    </div>
    <div class="mt-8 text-xs text-slate-500 border-t border-slate-800 pt-4">
      <strong>Methodology Note:</strong> Automated daily via GitHub Actions. Fundamental multiples pulled via Yahoo Finance NSE/BSE feeds. News headlines parsed via Google News RSS and scored using NLTK VADER lexicon (-1.0 to +1.0 polarity).
    </div>
  </div>
  <script>
    const stocks = {json_data};
    function renderKPIs() {{
      const total = stocks.length;
      const bullish = stocks.filter(s => s.news_mood === 'Bullish').length;
      const value = stocks.filter(s => s.val_status.includes('Value') || s.val_status.includes('Undervalued')).length;
      const actionable = stocks.filter(s => s.action.includes('Strong Buy') || s.action.includes('Momentum')).length;
      const kpis = [
        {{ label: 'Stocks Screened', val: total, sub: 'Diversified NSE Basket' }},
        {{ label: 'Bullish News Flow', val: bullish, sub: 'Positive VADER Polarity' }},
        {{ label: 'Value Opportunities', val: value, sub: 'P/E & Growth Discount' }},
        {{ label: 'Actionable Signals', val: actionable, sub: 'Valuation + Catalyst' }}
      ];
      document.getElementById('kpi-cards').innerHTML = kpis.map(k => `
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div class="text-slate-400 text-xs font-medium">${{k.label}}</div>
          <div class="text-2xl font-bold text-white mt-1">${{k.val}}</div>
          <div class="text-xs text-emerald-400 mt-1">${{k.sub}}</div>
        </div>
      `).join('');
    }}
    function renderTable() {{
      const search = document.getElementById('searchInput').value.toLowerCase();
      const mood = document.getElementById('moodFilter').value;
      const filtered = stocks.filter(s => {{
        const matchSearch = s.company.toLowerCase().includes(search) || s.sector.toLowerCase().includes(search) || s.ticker.toLowerCase().includes(search);
        const matchMood = mood === 'ALL' || s.news_mood === mood;
        return matchSearch && matchMood;
      }});
      document.getElementById('tableBody').innerHTML = filtered.map(s => {{
        const moodBadge = s.news_mood === 'Bullish'
          ? '<span class="px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">🟢 Bullish (' + s.sentiment_score + ')</span>'
          : s.news_mood === 'Bearish'
          ? '<span class="px-2.5 py-1 rounded-full text-xs font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20">🔴 Bearish (' + s.sentiment_score + ')</span>'
          : '<span class="px-2.5 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-300">⚪ Neutral (' + s.sentiment_score + ')</span>';
        return `
          <tr onclick="selectStock('${{s.company}}')" class="hover:bg-slate-800/50 cursor-pointer transition">
            <td class="py-3.5 px-4">
              <div class="font-semibold text-white">${{s.company}}</div>
              <div class="text-xs text-slate-400">${{s.ticker}} • ${{s.sector}}</div>
            </td>
            <td class="py-3.5 px-4 font-medium text-white">₹${{s.price}}</td>
            <td class="py-3.5 px-4 text-slate-300">₹${{s.high_52w}} <span class="text-xs text-slate-500">(${{s.pct_high}}%)</span></td>
            <td class="py-3.5 px-4 text-slate-300">${{s.pe}}</td>
            <td class="py-3.5 px-4 text-slate-300">${{s.val_status}}</td>
            <td class="py-3.5 px-4">${{moodBadge}}</td>
            <td class="py-3.5 px-4 font-medium text-emerald-300">${{s.action}}</td>
          </tr>
        `;
      }}).join('');
    }}
    function selectStock(companyName) {{
      document.getElementById('stockSelect').value = companyName;
      renderHeadlines();
    }}
    function renderHeadlines() {{
      const selected = document.getElementById('stockSelect').value;
      const stock = stocks.find(s => s.company === selected);
      if (!stock) return;
      if (!stock.headlines || stock.headlines.length === 0) {{
        document.getElementById('headlineContainer').innerHTML = '<p class="text-slate-400 text-sm">No recent news headlines found.</p>';
        return;
      }}
      document.getElementById('headlineContainer').innerHTML = stock.headlines.map(h => {{
        const color = h.score >= 0.15 ? 'text-emerald-400' : (h.score <= -0.15 ? 'text-rose-400' : 'text-slate-400');
        return `
          <div class="p-3.5 bg-slate-950/60 border border-slate-800 rounded-lg flex items-start justify-between gap-4">
            <a href="${{h.link}}" target="_blank" class="text-sm text-slate-200 hover:text-emerald-400 transition">${{h.title}}</a>
            <span class="text-xs font-mono font-bold ${{color}} shrink-0">Score: ${{h.score}}</span>
          </div>
        `;
      }}).join('');
    }}
    renderKPIs();
    renderTable();
    const selectEl = document.getElementById('stockSelect');
    selectEl.innerHTML = stocks.map(s => `<option value="${{s.company}}">${{s.company}} (${{s.ticker}})</option>`).join('');
    renderHeadlines();
    document.getElementById('searchInput').addEventListener('input', renderTable);
    document.getElementById('moodFilter').addEventListener('change', renderTable);
    selectEl.addEventListener('change', renderHeadlines);
  </script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_template)
