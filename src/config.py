import os

# 專案路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')

# 欄位定義 (必須與 Task 3 訓練時的順序一致)
NUMERIC_FEATURES = [
    "IRRADIATION", 
    "AMBIENT_TEMPERATURE", 
    "MODULE_TEMPERATURE",
    "hour_sin", 
    "hour_cos"
]

CATEGORICAL_FEATURES = ["SOURCE_KEY"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# 模型設定
PLANT_1 = "Plant 1"
PLANT_2 = "Plant 2"

TARGETS = ["DC_POWER", "AC_POWER"]

AVAILABLE_MODELS = [
    "Linear Regression",
    "Ridge",
    "Lasso",
    "Random Forest",
    "MLP (Neural Net)"
]

MODEL_FILENAME_MAP = {
    "Linear Regression": "linear",
    "Ridge": "ridge",
    "Lasso": "lasso",
    "Random Forest": "rf",
    "MLP (Neural Net)": "mlp"
}