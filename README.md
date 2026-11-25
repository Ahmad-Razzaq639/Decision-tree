# 📉 Telco Churn Studio (Decision Tree + Streamlit)

Multi-page Streamlit app built on your notebook's preprocessing and Decision Tree.

## Pages
| Page | What it does |
|---|---|
| 📊 Overview | KPIs, auto insights, churn by contract / internet / payment / tenure, correlations |
| 🗂️ Data Explorer | Filters, column profile, segment heatmap, data-quality checks, CSV download |
| 🧪 Model Lab | Live hyperparameters (criterion, max_depth, min_samples_*, class_weight, ccp_alpha, threshold, test size), metrics, confusion matrix, ROC/PR, tree plot, importance, overfitting sweep, 5-fold CV, model export |
| ⚖️ Compare Models | Your notebook's 6 configs + current model, ROC overlay, Gini vs Entropy, GridSearchCV (+ one-click apply) |
| 🔮 Predict | Single customer form, risk gauge, tree rules used, retention levers, what-if charts |
| 📦 Batch Predict | Upload CSV → scored file + risk summary + download |
| 📚 Concepts | Gini / Entropy / Information Gain calculators and the preprocessing table |

The model chosen in **Model Lab** is the one used by Predict and Batch Predict.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Push this whole folder to a GitHub repo (keep `Telco-Customer-Churn.csv` next to `app.py`).
2. Go to https://share.streamlit.io → **New app** → pick the repo, branch, and main file `app.py`.
3. Deploy. First load trains the models (a few seconds) and then everything is cached.

Other options: Hugging Face Spaces (SDK: Streamlit), Render, Railway, Fly.io, or Docker.

## Why not Vercel?
Streamlit needs a long-running Python server with a WebSocket connection. Vercel runs short-lived
serverless functions, so Streamlit apps do not run there. On Vercel you would need to rewrite the UI
in Next.js/React and host the model as a separate API (FastAPI) elsewhere or in a Vercel Python function.

## Notes
- Decision Trees don't need feature scaling, so the app skips `StandardScaler` (thresholds stay in real units).
  Tuned configs match the notebook exactly; only the fully-grown default tree can differ by ~0.1–0.8%
  because of tie-breaking among equal-gain splits.
- Batch CSV must use the original Telco columns (`customerID` and `Churn` optional).
