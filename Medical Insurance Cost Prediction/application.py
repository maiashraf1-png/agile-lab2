# application.py (train_model.py) - Medical Insurance Cost Prediction
# ---------------------------------------------------------
# Trains a Stochastic Gradient Boosting (SGB) Regressor on the
# "Medical Cost Personal" dataset and saves the model, scaler,
# and feature columns so app.py (Streamlit) can reuse them.
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------------------------------------------
# --- 1. Load and explore data ---
# -----------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load and explore data")
print("=" * 60)

df = pd.read_csv("insurance.csv")

print("\nShape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe(include="all"))

# Distribution of the target variable
plt.figure(figsize=(7, 4))
sns.histplot(df["charges"], kde=True, bins=30, color="#2a78d6")
plt.title("Distribution of Medical Charges")
plt.xlabel("Charges")
plt.tight_layout()
plt.savefig("charges_distribution.png", dpi=120)
plt.close()

# Correlation heatmap (numeric + encoded categorical features)
df_corr = df.copy()
df_corr["sex"] = df_corr["sex"].map({"male": 1, "female": 0})
df_corr["smoker"] = df_corr["smoker"].map({"yes": 1, "no": 0})
df_corr = pd.get_dummies(df_corr, columns=["region"], drop_first=True)

plt.figure(figsize=(8, 6))
sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120)
plt.close()

print("\nSaved EDA plots: charges_distribution.png, correlation_heatmap.png")
print("\nCorrelation with target (charges):\n",
      df_corr.corr(numeric_only=True)["charges"].sort_values(ascending=False))

# -----------------------------------------------------------------
# --- 2. Handle missing values (treat zeros as missing) ---
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Handle missing values (treat zeros as missing)")
print("=" * 60)

# Columns where a value of 0 is physically impossible / a data-entry
# error (unlike 'children', where 0 is a perfectly valid real value).
zero_as_missing_cols = ["age", "bmi"]

for col in zero_as_missing_cols:
    n_zeros = (df[col] == 0).sum()
    if n_zeros > 0:
        df.loc[df[col] == 0, col] = np.nan
        print(f"Column '{col}': {n_zeros} zero value(s) converted to NaN")
    else:
        print(f"Column '{col}': no zero values found")

# Report any real NaNs (from the dataset itself or from the conversion above)
print("\nMissing values BEFORE imputation:\n", df.isnull().sum())

# Impute numeric columns with median (robust to the right-skewed charges,
# and to outliers in bmi/age), categorical columns with mode
num_cols_with_na = df.select_dtypes(include=[np.number]).columns[
    df.select_dtypes(include=[np.number]).isnull().any()
]
for col in num_cols_with_na:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"Filled missing '{col}' with median = {median_val:.2f}")

cat_cols_with_na = df.select_dtypes(include=["object", "str"]).columns[
    df.select_dtypes(include=["object", "str"]).isnull().any()
]
for col in cat_cols_with_na:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"Filled missing '{col}' with mode = {mode_val}")

print("\nMissing values AFTER imputation:\n", df.isnull().sum())

# --- Encode categorical features ---
print("\nEncoding categorical features (sex, smoker: label encoding | region: one-hot encoding)")
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df = pd.get_dummies(df, columns=["region"], drop_first=True)

# Make sure the one-hot columns are proper 0/1 ints (not True/False)
region_dummy_cols = [c for c in df.columns if c.startswith("region_")]
df[region_dummy_cols] = df[region_dummy_cols].astype(int)

print("\nEncoded dataframe preview:\n", df.head())

# -----------------------------------------------------------------
# --- 3. Define features and target ---
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Define features and target")
print("=" * 60)

X = df.drop(columns=["charges"])
y = df["charges"]

feature_columns = X.columns.tolist()
print("\nFeature columns:", feature_columns)
print("Target column: charges")

# -----------------------------------------------------------------
# --- 4. Scale features (fit on full data so the UI can use the same scaler) ---
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Scale features")
print("=" * 60)

continuous_cols = ["age", "bmi", "children"]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[continuous_cols] = scaler.fit_transform(X[continuous_cols])

print("Continuous columns normalized with StandardScaler:", continuous_cols)
print("\nScaled feature preview:\n", X_scaled.head())

# -----------------------------------------------------------------
# --- 5. Split, train, evaluate ---
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Split, train, evaluate")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Stochastic Gradient Boosting: GradientBoostingRegressor with
# subsample < 1.0 makes it "stochastic" (each tree trains on a random
# subsample of the training data, which reduces variance/overfitting).
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,      # <-- the "stochastic" part
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nR2 Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"MSE      : {mse:.2f}")
print(f"RMSE     : {rmse:.2f}")

# --- Feature importance ---
importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
print("\nFeature importance:\n", importances)

plt.figure(figsize=(7, 4))
sns.barplot(x=importances.values, y=importances.index, color="#2a78d6")
plt.title("Feature Importance (SGB model)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()
print("Saved plot: feature_importance.png")

# -----------------------------------------------------------------
# --- 6. Save the model, scaler, and feature columns ---
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Save the model, scaler, and feature columns")
print("=" * 60)

joblib.dump(model, "sgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
joblib.dump(continuous_cols, "continuous_cols.pkl")

# Save metrics too, so the Streamlit app can display them without retraining
metrics = {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse}
joblib.dump(metrics, "metrics.pkl")

print("\nSaved: sgb_model.pkl, scaler.pkl, feature_columns.pkl, continuous_cols.pkl, metrics.pkl")
print("\nDone. You can now run: streamlit run app.py")