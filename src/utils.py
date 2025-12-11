# src/utils.py
import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st
import config

@st.cache_resource
def load_model(plant_name, target, model_name):
    """
    載入訓練好的模型 Pipeline (.pkl)
    根據使用者選擇的 plant, target 和 model
    """
    try:
        # 取得模型檔名前綴
        model_prefix = config.MODEL_FILENAME_MAP.get(model_name)
        if not model_prefix:
            st.error(f"Unknown model: {model_name}")
            return None
        
        # 建構檔名
        plant_num = "1" if plant_name == config.PLANT_1 else "2"
        target_lower = target.lower().replace("_power", "")
        filename = f"plant{plant_num}_{target_lower}_{model_prefix}.pkl"
        
        path = os.path.join(config.MODEL_DIR, filename)
        
        # 檢查檔案是否存在
        if not os.path.exists(path):
            st.error(f"Model file not found: {filename}")
            return None
            
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model {filename}: {e}")
        return None

def preprocess_input(irradiation, amb_temp, mod_temp, hour, source_key):
    """
    將使用者輸入轉換為模型可接受的 DataFrame 格式
    包含 Task 3 中的 Cyclic Encoding (hour_sin, hour_cos)
    """
    # 時間特徵的循環編碼 (Cyclic Encoding)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    data = {
        "IRRADIATION": [irradiation],
        "AMBIENT_TEMPERATURE": [amb_temp],
        "MODULE_TEMPERATURE": [mod_temp],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
        "SOURCE_KEY": [source_key]
    }
    
    # 建立 DataFrame，欄位順序必須與 config 中定義的一致
    df = pd.DataFrame(data, columns=config.ALL_FEATURES)
    return df

