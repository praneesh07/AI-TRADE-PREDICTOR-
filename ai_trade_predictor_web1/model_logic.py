import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def predict_trade(user_date_str):
 #  try:
        # Load and preprocess data
        df = pd.read_excel("model/financial_data.xlsx")
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        df['Price Change'] = df['Close'] - df['Open']
        df['Volatility'] = df['High'] - df['Low']
        df['Target'] = df['Price Change'].apply(lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0))

        # Features and target
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Price Change', 'Volatility']]
        y = df['Target']

        if len(df) < 10:
            return "Not enough data to make prediction."

        # Train model
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = RandomForestClassifier()
        model.fit(X_train_scaled, y_train)     

        # Prepare prediction date
        user_date = pd.to_datetime(user_date_str)
        last_year_date = user_date - pd.DateOffset(years=1)   

        # Find the closest available historical date
        df['DateDiff'] = (df['Date'] - last_year_date).abs() 
        matched_row = df.loc[df['DateDiff'].idxmin()]

        if abs((matched_row['Date'] - last_year_date).days) > 3:
            return f"No historical data close to {last_year_date.date()}"

        user_features = matched_row[['Open', 'High', 'Low', 'Close', 'Volume', 'Price Change', 'Volatility']]
        user_features_scaled = scaler.transform([user_features])
        prediction = model.predict(user_features_scaled)[0]
        close_price = matched_row['Close']

        label = {1: "BUY", 0: "HOLD", -1: "SELL"}
        return f"{label[prediction]} at ₹{close_price:.2f} (based on {matched_row['Date'].date()})"
    
   
