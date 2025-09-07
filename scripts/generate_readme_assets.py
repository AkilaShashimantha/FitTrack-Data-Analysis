import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'cleaned_fitness_data.csv')
OUT_DIR = os.path.join(ROOT, 'docs')
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Feature set used in the app / notebook
FEATURES = [
    'TotalSteps', 'TotalDistance', 'VeryActiveMinutes', 'FairlyActiveMinutes',
    'LightlyActiveMinutes', 'SedentaryMinutes', 'TotalMinutesAsleep'
]
TARGET = 'Calories'

df = df.dropna(subset=FEATURES + [TARGET]).copy()
X = df[FEATURES].values
y = df[TARGET].values

# Train/test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
}

r2_scores = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2_scores[name] = r2_score(y_test, y_pred)

# Model comparison bar chart
names = list(r2_scores.keys())
scores = [r2_scores[n] for n in names]
order = np.argsort(scores)[::-1]

plt.figure(figsize=(9, 5))
plt.bar(np.array(names)[order], np.array(scores)[order], color=['#1E88E5']*len(names))
plt.ylabel('R-squared')
plt.ylim(0, 1)
plt.title('Model Performance Comparison (R-squared)')
for i, v in enumerate(np.array(scores)[order]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'model_comparison.png'), dpi=200)
plt.close()

# Correlation heatmap (subset)
corr = df[FEATURES + [TARGET]].corr(numeric_only=True)
plt.figure(figsize=(8, 6))
im = plt.imshow(corr, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr.index)), corr.index)
plt.title('Correlation Heatmap of Key Features')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'corr_heatmap.png'), dpi=200)
plt.close()

print('Assets written to:', OUT_DIR)
