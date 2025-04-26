import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt
import json
import os
import yfinance as yf
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

# ----------------- SETTINGS -----------------

TOP_50 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "XOM", "UNH", "JPM",
    "JNJ", "V", "PG", "LLY", "MA", "AVGO", "HD", "CVX", "MRK", "PEP",
    "ABBV", "COST", "KO", "WMT", "ADBE", "CRM", "MCD", "BAC", "NFLX", "ABT",
    "ACN", "TMO", "PFE", "LIN", "DHR", "DIS", "VZ", "BMY", "WFC", "TXN",
    "INTC", "AMGN", "NEE", "UNP", "LOW", "MS", "HON", "QCOM", "ORCL", "IBM"
]

TOP_100 = TOP_50 + [
    "AMD", "GS", "PM", "RTX", "NOW", "CAT", "ISRG", "SCHW", "AMAT", "BLK",
    "GE", "SPGI", "SBUX", "PLD", "INTU", "MDT", "SYK", "T", "ADI", "LRCX",
    "DE", "MU", "BKNG", "MMC", "ZTS", "TJX", "GILD", "EL", "ADP", "CB",
    "USB", "C", "CI", "REGN", "VRTX", "BDX", "PNC", "CSCO", "EW", "NSC",
    "SO", "CL", "FDX", "PGR", "APD", "GM", "PSX", "ITW", "AON", "HUM"
]

# ----------------- METADATA MANAGEMENT -----------------

def load_metadata():
    if not os.path.exists("metadata.json"):
        metadata = {
            "date": datetime.today().strftime("%Y-%m-%d"),
            "db_version": 1,
            "ml_version": 1,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        save_metadata(metadata)
    else:
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
    return metadata

def save_metadata(metadata):
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

metadata = load_metadata()

# ----------------- STREAMLIT SETUP -----------------

st.set_page_config(page_title="📈 Stock Analyzer Pro Final", layout="wide")
st.title("🏆 Stock Analyzer Pro Final")

# Display Version
version_string = f"v{metadata['date'].replace('-', '.')}.DB{metadata['db_version']}.ML{metadata['ml_version']}"
st.caption(f"Version: {version_string} | Last Updated: {metadata['last_updated']}")

# ----------------- BUTTONS: BUILD DATASET / RETRAIN MODEL -----------------

st.sidebar.header("🏗️ Dataset and Model Management")

dataset_mode = st.sidebar.radio("Choose Dataset Build Mode:", ("Light (50 stocks)", "Full (100 stocks)"))

if st.sidebar.button("Build Historical Dataset"):
    with st.spinner("Building historical dataset... This may take a few minutes..."):
        stock_list = TOP_50 if dataset_mode == "Light (50 stocks)" else TOP_100
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="7y")["Close"]
        start_date = (datetime.today() - timedelta(days=365*7)).strftime('%Y-%m-%d')
        data_records = []

        for ticker in stock_list:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date)

                if len(hist) < 500:
                    continue

                ref_date = hist.index[0] + timedelta(days=365*3)
                hist_ref = hist[hist.index <= ref_date]

                if hist_ref.empty:
                    continue

                info = stock.info
                revenue_growth = info.get('revenueGrowth')
                profit_margins = info.get('profitMargins')
                debt_equity = info.get('debtToEquity')
                dividend_yield = info.get('dividendYield')

                try:
                    dividends = stock.dividends
                    dividend_growth = dividends.pct_change().mean() if not dividends.empty else 0
                except:
                    dividend_growth = 0

                try:
                    price_momentum = (hist["Close"].iloc[-1] - hist["Close"].iloc[-252]) / hist["Close"].iloc[-252]
                except:
                    price_momentum = None

                future_date = ref_date + timedelta(days=365*3)
                future_prices = hist[hist.index >= future_date]["Close"]

                if future_prices.empty:
                    continue

                future_price = future_prices.iloc[0]
                start_price = hist_ref["Close"].iloc[-1]

                stock_3y_return = ((future_price / start_price) ** (1/3)) - 1 if start_price and future_price else None

                spy_ref_price = spy_hist[spy_hist.index <= ref_date].iloc[-1]
                spy_future_prices = spy_hist[spy_hist.index >= future_date]

                if spy_future_prices.empty:
                    continue

                spy_future_price = spy_future_prices.iloc[0]
                spy_3y_return = ((spy_future_price / spy_ref_price) ** (1/3)) - 1 if spy_ref_price and spy_future_price else None

                label = 1 if stock_3y_return and spy_3y_return and stock_3y_return > spy_3y_return else 0

                data_records.append({
                    "Ticker": ticker,
                    "Revenue Growth %": revenue_growth * 100 if revenue_growth else None,
                    "Profit Margin %": profit_margins * 100 if profit_margins else None,
                    "Debt/Equity": debt_equity,
                    "Dividend Yield %": dividend_yield * 100 if dividend_yield else None,
                    "Dividend Growth %": dividend_growth * 100 if dividend_growth else None,
                    "1Y Price Momentum %": price_momentum * 100 if price_momentum else None,
                    "Label (Outperform?)": label
                })

                time.sleep(0.5)
            except:
                continue

        df_full = pd.DataFrame(data_records)
        df_full.dropna(inplace=True)

        if "Label (Outperform?)" not in df_full.columns or df_full["Label (Outperform?)"].nunique() < 2:
            st.error("⚠️ Dataset did not have enough diversity (both outperform and underperform). Try again.")
        else:
            df_full.to_csv("historical_stock_dataset.csv", index=False)
            metadata["date"] = datetime.today().strftime("%Y-%m-%d")
            metadata["db_version"] += 1
            metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            save_metadata(metadata)
            st.success("✅ Dataset built and saved.")

if st.sidebar.button("Retrain ML Model"):
    with st.spinner("Training models..."):

        df = pd.read_csv("historical_stock_dataset.csv")
        X = df[["Revenue Growth %", "Profit Margin %", "Debt/Equity", "Dividend Yield %", "Dividend Growth %", "1Y Price Momentum %"]]
        y = df["Label (Outperform?)"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 1️⃣ Train Random Forest
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
        rf_model.fit(X_train, y_train)
        rf_accuracy = rf_model.score(X_test, y_test)

        with open("stock_model_rf.pkl", "wb") as f:
            pickle.dump(rf_model, f)

        # 2️⃣ Train XGBoost
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_accuracy = xgb_model.score(X_test, y_test)

        with open("stock_model_xgb.pkl", "wb") as f:
            pickle.dump(xgb_model, f)

        # 3️⃣ Pick best model
        if xgb_accuracy >= rf_accuracy:
            best_model = xgb_model
            best_model_name = "XGBoost"
            best_accuracy = xgb_accuracy
        else:
            best_model = rf_model
            best_model_name = "Random Forest"
            best_accuracy = rf_accuracy

        with open("stock_model_final.pkl", "wb") as f:
            pickle.dump(best_model, f)

        # 4️⃣ Update metadata
        metadata["ml_version"] += 1
        metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_metadata(metadata)

    st.success(f"✅ Models trained. Best model: {best_model_name} ({best_accuracy*100:.2f}% accuracy) saved as stock_model_final.pkl!")


#end Part 1

#start Part 2

# ----------------- MAIN ANALYSIS -----------------

st.sidebar.header("⚙️ Stock Analysis")

profile = st.sidebar.selectbox("Investor Style:", ["Balanced", "Conservative", "Growth"])
rank_method = st.sidebar.selectbox("Rank Stocks By:", ["Manual Score", "ML Prediction", "Blended Score"])

# Load data
if os.path.exists("historical_stock_dataset.csv") and os.path.exists("stock_model.pkl"):
    df = pd.read_csv("historical_stock_dataset.csv")
    with open("stock_model_final.pkl", "rb") as f:
        model = pickle.load(f)

    def manual_score(row):
        score = 0
        if profile == "Balanced":
            if row["Revenue Growth %"] > 5: score += 2
            if row["Profit Margin %"] > 10: score += 2
            if row["Debt/Equity"] < 100: score += 2
            if row["Dividend Yield %"] > 1: score += 2
            if row["1Y Price Momentum %"] > 5: score += 2
        elif profile == "Conservative":
            if row["Profit Margin %"] > 15: score += 3
            if row["Dividend Yield %"] > 1.5: score += 3
            if row["Debt/Equity"] < 50: score += 2
        elif profile == "Growth":
            if row["Revenue Growth %"] > 10: score += 4
            if row["1Y Price Momentum %"] > 10: score += 4
        return score

    # Apply manual scoring
    df["Manual Score"] = df.apply(manual_score, axis=1)

    # Apply ML predictions safely
    features = ["Revenue Growth %", "Profit Margin %", "Debt/Equity", "Dividend Yield %", "Dividend Growth %", "1Y Price Momentum %"]

    if len(model.classes_) == 2:
        df["ML Outperform Probability %"] = model.predict_proba(df[features])[:, 1] * 100
    else:
        df["ML Outperform Probability %"] = model.predict_proba(df[features])[:, 0] * 100

    # Blend
    df["Blended Score"] = df["Manual Score"] * 10 + df["ML Outperform Probability %"]

    # Sorting
    if rank_method == "Manual Score":
        sorted_df = df.sort_values(by="Manual Score", ascending=False)
    elif rank_method == "ML Prediction":
        sorted_df = df.sort_values(by="ML Outperform Probability %", ascending=False)
    else:
        sorted_df = df.sort_values(by="Blended Score", ascending=False)

    # ----------------- DISPLAY -----------------

    st.subheader("📋 Stock Rankings")

    st.dataframe(sorted_df[["Ticker", "Manual Score", "ML Outperform Probability %", "Blended Score"]].reset_index(drop=True))

    # 📊 Charts

    st.subheader("📊 Top 10 by Manual Score")
    top_manual = sorted_df.sort_values(by="Manual Score", ascending=False).head(10)
    manual_chart = alt.Chart(top_manual).mark_bar().encode(
        x=alt.X('Ticker:N', sort='-y'),
        y='Manual Score:Q',
        color='Ticker:N'
    ).properties(width=700, height=400)
    st.altair_chart(manual_chart)

    st.subheader("📊 Top 10 by ML Prediction")
    top_ml = sorted_df.sort_values(by="ML Outperform Probability %", ascending=False).head(10)
    ml_chart = alt.Chart(top_ml).mark_bar().encode(
        x=alt.X('Ticker:N', sort='-y'),
        y='ML Outperform Probability %:Q',
        color='Ticker:N'
    ).properties(width=700, height=400)
    st.altair_chart(ml_chart)

    # 📥 Download CSV
    csv = sorted_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Results", data=csv, file_name="full_stock_analysis.csv", mime="text/csv")

    # ----------------- EMAIL BEST PICKS -----------------

    st.sidebar.header("✉️ Email Best Picks")

    send_email = st.sidebar.checkbox("Send Email")

    if send_email:
        sender = st.sidebar.text_input("Sender Gmail")
        password = st.sidebar.text_input("Sender Password", type="password")
        receiver = st.sidebar.text_input("Receiver Email")

        if st.sidebar.button("Send Now"):
            best_manual = sorted_df.iloc[0]["Ticker"]
            best_ml = sorted_df.sort_values(by="ML Outperform Probability %", ascending=False).iloc[0]["Ticker"]

            subject = "📈 Weekly Stock Best Picks"
            body = f"🏆 Best by Manual Score: {best_manual}\n🤖 Best by ML Prediction: {best_ml}"

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = receiver

            try:
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.login(sender, password)
                server.sendmail(sender, receiver, msg.as_string())
                server.quit()
                st.success("✅ Email Sent!")
            except Exception as e:
                st.error(f"❌ Email Failed: {e}")

else:
    st.warning("⚠️ Please build the historical dataset and train the ML model first!")

