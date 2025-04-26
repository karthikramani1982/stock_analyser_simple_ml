# stock_analyser_simple_ml
stock analyzer and ML based prediction of best performing stocks over the next 3 years
Stock Analyzer is a fully interactive, machine-learning-enhanced stock screening and ranking app built with Streamlit.  
It allows you to:

- Build a historical stock dataset (Light Mode or Full Mode)
- Train and retrain a machine learning model
- Score stocks manually based on growth/conservative/balanced profiles
- Predict outperformers using ML probabilities
- Visualize Top 10 stocks via charts
- Download full stock analysis
- Email yourself the weekly top picks
- Track app and database versioning automatically

---

## 🚀 Features

- **Database Build**  
  - Pulls real financial and price data from Yahoo Finance
  - Supports Light Mode (50 stocks) or Full Mode (100 stocks)

- **Machine Learning**  
  - Auto-retraining inside the app with 1 click
  - Random Forest Classifier to predict 3-year outperformers
  - XGboost algorithm to also predict 3-year outperformers
  - Automatically selects best model (Random Forest or XGBoost) during retraining.

- **Stock Scoring and Ranking**  
  - Manual scoring based on Revenue Growth, Margins, Debt, Dividends, Momentum
  - Machine Learning prediction of outperform probabilities
  - Blended scoring (Manual + ML)

- **Charts and Visualization**  
  - Top 10 stocks by Manual Score and ML Prediction

- **Email Alerts**  
  - Send yourself the weekly best picks via email directly from the app

- **Professional Versioning System**  
  - Tracks App Version as `vYYYY.MM.DD.DB.ML` (Database version and ML model version separately)

---

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-analyzer.git
   cd stock-analyzer

Install requirements as follows:

pip install -r requirements.txt

To run a local build: 

streamlit run stock_analyzer_bulk_pro_final.py

