"""
Price Prediction Model using Machine Learning
ทำนายราคาตลาดของนักเตะจากสถิติโดยใช้ Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class PlayerPricePredictor:
    """โมเดลสำหรับทำนายราคานักเตะ"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'Finishing', 'Positioning', 'xG', 'Speed', 'Strength',
            'Passing', 'PressActions', 'Vision', 'Aggression',
            'Composure', 'OffTheBall'
        ]

    def prepare_features(self, df):
        """เตรียมข้อมูล features"""
        # Handle missing values
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 75  # ค่าเฉลี่ยถ้าไม่มีข้อมูล

        X = df[self.feature_columns].copy()

        # Feature Engineering
        # 1. Goal Threat Score
        X['GoalThreat'] = (df['Finishing'] * 0.5 + df['Positioning'] * 0.3 + df['xG'] * 100 * 0.2)

        # 2. Physical Score
        X['Physical'] = (df['Speed'] * 0.5 + df['Strength'] * 0.5)

        # 3. Technical Score
        X['Technical'] = (df['Passing'] * 0.4 + df['Vision'] * 0.3 + df['Composure'] * 0.3)

        # 4. Work Rate Score
        X['WorkRate'] = (df['PressActions'] * 10 + df['Aggression'] * 0.5 + df['OffTheBall'] * 0.5)

        return X

    def train(self, df_with_prices):
        """เทรนโมเดล"""
        # Prepare data
        X = self.prepare_features(df_with_prices)
        y = df_with_prices['MarketValue']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ Model trained successfully!")
        print(f"📊 MAE: €{mae/1_000_000:.2f}M")
        print(f"📊 R² Score: {r2:.3f}")

        return mae, r2

    def predict(self, df):
        """ทำนายราคา"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Add predictions to dataframe
        df_result = df.copy()
        df_result['PredictedPrice'] = predictions
        df_result['PredictedPrice_M'] = (predictions / 1_000_000).round(1)

        return df_result

    def get_feature_importance(self):
        """ดูความสำคัญของแต่ละ feature"""
        if self.model is None:
            return None

        importance_df = pd.DataFrame({
            'Feature': self.feature_columns + ['GoalThreat', 'Physical', 'Technical', 'WorkRate'],
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        return importance_df

    def save_model(self, filepath='models/price_predictor.pkl'):
        """บันทึกโมเดล"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.feature_columns
        }, filepath)
        print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath='models/price_predictor.pkl'):
        """โหลดโมเดล"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['features']
        print(f"✅ Model loaded from {filepath}")


def train_and_save_model(df_with_market_values):
    """
    Train model และบันทึก
    df_with_market_values ต้องมีคอลัมน์ MarketValue
    """
    predictor = PlayerPricePredictor()
    mae, r2 = predictor.train(df_with_market_values)

    # Save model
    import os
    os.makedirs('models', exist_ok=True)
    predictor.save_model()

    return predictor, mae, r2


def predict_prices(df, model_path='models/price_predictor.pkl'):
    """ใช้โมเดลที่เทรนแล้วทำนายราคา"""
    predictor = PlayerPricePredictor()

    try:
        predictor.load_model(model_path)
    except:
        print("⚠️ Model not found, using statistical estimation instead")
        # Fallback to simple estimation
        return estimate_prices_simple(df)

    return predictor.predict(df)


def estimate_prices_simple(df):
    """
    วิธีประมาณราคาแบบง่าย (fallback)
    ใช้เมื่อไม่มีโมเดล ML
    """
    df_result = df.copy()

    # Calculate price based on key stats
    goal_threat = (df['Finishing'] * 0.4 + df['Positioning'] * 0.3 + df['xG'] * 100 * 0.3)
    physical = (df['Speed'] + df['Strength']) / 2
    technical = (df['Passing'] + df.get('Vision', 75)) / 2

    # Overall score
    overall_score = (goal_threat * 0.5 + physical * 0.3 + technical * 0.2)

    # Price tiers
    base_price = 1_000_000

    def calculate_price(score):
        if score >= 90:
            return base_price * 150  # 150M+
        elif score >= 85:
            return base_price * 80   # 80M
        elif score >= 80:
            return base_price * 40   # 40M
        elif score >= 75:
            return base_price * 20   # 20M
        elif score >= 70:
            return base_price * 10   # 10M
        else:
            return base_price * 5    # 5M

    df_result['PredictedPrice'] = overall_score.apply(calculate_price)
    df_result['PredictedPrice_M'] = (df_result['PredictedPrice'] / 1_000_000).round(1)

    return df_result


# Test
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/players.csv")

    # Add market values (from transfermarkt module)
    from transfermarkt_scraper import add_market_values_to_dataframe
    df = add_market_values_to_dataframe(df)

    # Train model
    print("🎯 Training Price Prediction Model...")
    predictor, mae, r2 = train_and_save_model(df)

    # Show predictions
    df_pred = predictor.predict(df.head(10))
    print("\n📊 Top 10 Players - Price Predictions:")
    print(df_pred[['Player', 'MarketValue_M', 'PredictedPrice_M']].to_string(index=False))

    # Feature importance
    print("\n📈 Feature Importance:")
    print(predictor.get_feature_importance().head(10))
