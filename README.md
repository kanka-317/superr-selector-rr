# SupeRR Selector Season 2: Interactive Pressure Reliability Dashboard
### Official Submission — Rajasthan Royals (RR)

Live interactive dashboard to explore and verify the Top 10 Batters and Top 10 Bowlers under pressure in T20 cricket (2022–2026), built with **Streamlit** and **Plotly**.

---

## 🚀 Free Deployment Guide (Streamlit Community Cloud)

Deploying this dashboard for free takes less than 2 minutes using Streamlit Community Cloud (free tier):

### Step 1: Create a GitHub Repository
1. Open your terminal in this `dashboard/` directory (or push this directory as a repository):
   ```bash
   cd dashboard
   git init
   git add .
   git commit -m "Initial commit: SupeRR Selector RR Dashboard"
   ```
2. Go to [GitHub.com](https://github.com/) and create a new **Public** repository (e.g. `superr-selector-rr`).
3. Link your remote repository and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/kanka-317/superr-selector-rr.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Community Cloud (Free)
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click **"New app"**.
3. Fill in the repository details:
   - **Repository:** `kanka-317/superr-selector-rr`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. (Optional) Under **Advanced settings**, ensure Python 3.10+ is selected.
5. Click **"Deploy!"**.

Your application will build in ~45 seconds and be accessible worldwide via a public URL such as:
`https://<YOUR_APP_NAME>.streamlit.app`

> ⚠️ **Free Tier Inactivity Note:** Streamlit Cloud's free tier automatically puts apps to sleep after 7 days of inactivity. Please open the link once before evaluation so the container wakes up instantly!

---

## 💻 Running Locally

To run the interactive dashboard on your local machine:

1. Clone or navigate into this folder:
   ```bash
   cd dashboard
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit server:
   ```bash
   streamlit run app.py
   ```
4. The dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## 📊 Dashboard Pages & Features

1. **🏆 Top 10 Rankings:**
   - Headline metrics and interactive Top 10 tables for Batters and Bowlers.
   - Exact numerical alignment with `rankings.xlsx` and the methodology PDF.
   - One-click **Download Top 10 CSV** export button.
2. **📈 Pressure Index Explorer:**
   - Distribution of the Continuous Pressure Index (0 to 1).
   - Dynamic **"What counts as pressure?"** threshold slider showing how game-state separation alters across the 60th to 90th percentiles.
3. **👤 Player Deep Dive:**
   - Interactive profile card for any of the 202 eligible players.
   - Situational deltas vs player's own baseline (Strike Rate & Economy).
   - Phase breakdown (Powerplay, Middle, Death) and season-by-season stability.
   - **Rajasthan Royals Tactical Fit** scouting notes (role, strengths, vulnerabilities).
4. **⚖️ Compare Players:**
   - Multi-select side-by-side radar fingerprint across 5 standardized dimensions.
   - Comparative head-to-head metric table.
5. **🧪 Model Robustness:**
   - Weight sensitivity simulation across 500 Dirichlet random perturbations.
   - Split-half reliability metrics ($r=0.684$ for batters, $r=0.712$ for bowlers).
   - Supervised Win Probability (WP) model validation metrics.
6. **📖 Methodology & PDF:**
   - Mathematical summaries of the Pressure Index, baseline formulation, and Empirical Bayes shrinkage.
   - In-app preview and download button for the official 14-page methodology report.

---

## 📦 Data Architecture & Lightweight Footprint

To ensure lightning-fast cloud loading and adhere to strict data privacy:
- The proprietary 201k-row raw ball-by-ball dataset is **excluded** from the repository (`.gitignore`).
- All metrics are precomputed into lightweight summary artifacts in `/data` (< 0.3 MB total).
- The dashboard utilizes Streamlit's `@st.cache_data` for sub-second page transitions.

