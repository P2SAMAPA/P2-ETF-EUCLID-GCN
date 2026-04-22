"""
Streamlit Dashboard for P2-ETF-EUCLID-GCN Engine.
"""

import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Euclid GCN", page_icon="🕸️", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .hero-return { font-size: 2rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.startswith("euclid_gcn_") and f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🕸️ P2Quant Euclid GCN</div>', unsafe_allow_html=True)
st.markdown('<div>Euclidean Graph Convolutional Network – ETF‑Macro Relational Learning</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available. Please run the daily pipeline first.")
    st.stop()

daily = data['daily_trading']
universes = daily['universes']
top_picks = daily['top_picks']

tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        top = top_picks.get(key)
        top3 = universes.get(key, [])
        if top:
            ticker = top['ticker']
            ret = top['predicted_return']
            st.markdown(f"""
            <div class="hero-card">
                <div style="font-size: 1.2rem; opacity: 0.8;">🕸️ TOP PICK</div>
                <div class="hero-ticker">{ticker}</div>
                <div class="hero-return">Predicted Return: {ret*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Top 3 Predictions")
            df = pd.DataFrame(top3)
            df['Predicted Return'] = df['predicted_return'].apply(lambda x: f"{x*100:.2f}%")
            df = df[['ticker', 'Predicted Return']].rename(columns={'ticker': 'Ticker'})
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No predictions for {key}.")
