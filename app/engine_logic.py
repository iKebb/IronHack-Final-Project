# engine_logic.py
import pandas as pd
import numpy as np
import pickle
import warnings
from typing import Tuple, Optional
import os
from config import COLORS, SENSOR_INFO

warnings.filterwarnings('ignore')

class EngineAnalyzer:
  def __init__(self):
    self.model = None
    self.sensor_info = SENSOR_INFO
    self.model_loaded = False
    self.expected_features = None
    
  def load_model(self, model_path="../models/trained_model.pkl"):
    try:
      possible_paths = [
        model_path,
        "models/trained_model.pkl",
        "../models/trained_model.pkl",
        os.path.join(os.path.dirname(__file__), "../models/trained_model.pkl")
      ]
      
      for path in possible_paths:
        if os.path.exists(path):
          with open(path, 'rb') as f:
            self.model = pickle.load(f)
          self.model_loaded = True
          self._extract_expected_features()
          return True
      
      print("Model file not found. Using heuristic estimation.")
      self.model_loaded = False
      return False
      
    except Exception as e:
      print(f"Warning: Could not load model: {e}")
      self.model_loaded = False
      return False
  
  def _extract_expected_features(self):
    try:
      if hasattr(self.model, 'feature_names_in_'):
        self.expected_features = list(self.model.feature_names_in_)
      elif hasattr(self.model, 'get_booster'):
        self.expected_features = self.model.get_booster().feature_names
      elif hasattr(self.model, 'feature_name_'):
        self.expected_features = self.model.feature_name_
      else:
        self.expected_features = [
          'sensor_measure_2', 'sensor_measure_3', 'sensor_measure_4', 
          'sensor_measure_6', 'sensor_measure_7', 'sensor_measure_8', 
          'sensor_measure_9', 'sensor_measure_11', 'sensor_measure_12', 
          'sensor_measure_13', 'sensor_measure_14', 'sensor_measure_15', 
          'sensor_measure_17', 'sensor_measure_20', 'sensor_measure_21',
          'T30_norm', 'd_T30', 'Ps30_norm', 'd_Ps30', 'T50_norm', 'd_T50',
          'thermal_stress', 'pressure_ratio'
        ]
    except Exception as e:
      print(f"Could not extract expected features: {e}")
      self.expected_features = None
  
  def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    
    for unit in X["unit"].unique():
      unit_mask = X["unit"] == unit
      
      if "sensor_measure_3" in X.columns:
        initial_T30 = X.loc[unit_mask, "sensor_measure_3"].iloc[0]
        if initial_T30 != 0:
          X.loc[unit_mask, "T30_norm"] = X.loc[unit_mask, "sensor_measure_3"] / initial_T30
          X.loc[unit_mask, "d_T30"] = X.loc[unit_mask, "T30_norm"].diff().fillna(0)
      
      if "sensor_measure_11" in X.columns:
        initial_Ps30 = X.loc[unit_mask, "sensor_measure_11"].iloc[0]
        if initial_Ps30 != 0:
          X.loc[unit_mask, "Ps30_norm"] = X.loc[unit_mask, "sensor_measure_11"] / initial_Ps30
          X.loc[unit_mask, "d_Ps30"] = X.loc[unit_mask, "Ps30_norm"].diff().fillna(0)
      
      if "sensor_measure_4" in X.columns:
        initial_T50 = X.loc[unit_mask, "sensor_measure_4"].iloc[0]
        if initial_T50 != 0:
          X.loc[unit_mask, "T50_norm"] = X.loc[unit_mask, "sensor_measure_4"] / initial_T50
          X.loc[unit_mask, "d_T50"] = X.loc[unit_mask, "T50_norm"].diff().fillna(0)
    
    if "sensor_measure_4" in X.columns and "sensor_measure_3" in X.columns:
      X["thermal_stress"] = X["sensor_measure_4"] - X["sensor_measure_3"]
    
    if "sensor_measure_11" in X.columns and "sensor_measure_7" in X.columns:
      valid_mask = X["sensor_measure_7"] != 0
      X.loc[valid_mask, "pressure_ratio"] = X.loc[valid_mask, "sensor_measure_11"] / X.loc[valid_mask, "sensor_measure_7"]
      X["pressure_ratio"] = X["pressure_ratio"].fillna(0)
    
    enhanced_format_columns = [
      'unit',
      'sensor_measure_2', 'sensor_measure_3', 'sensor_measure_4',
      'sensor_measure_6', 'sensor_measure_7', 'sensor_measure_8',
      'sensor_measure_9', 'sensor_measure_11', 'sensor_measure_12',
      'sensor_measure_13', 'sensor_measure_14', 'sensor_measure_15',
      'sensor_measure_17', 'sensor_measure_20', 'sensor_measure_21',
      'T30_norm', 'd_T30', 'Ps30_norm', 'd_Ps30', 'T50_norm', 'd_T50',
      'thermal_stress', 'pressure_ratio'
    ]
    
    existing_columns = [col for col in enhanced_format_columns if col in X.columns]
    
    for col in enhanced_format_columns:
      if col not in X.columns and col not in ['unit']:
        X[col] = 0
    
    final_columns = [col for col in enhanced_format_columns if col in X.columns]
    return X[final_columns]
  
  def _align_features_with_model(self, features: pd.DataFrame) -> pd.DataFrame:
    if self.expected_features is None:
      return features
    
    aligned_features = pd.DataFrame()
    for feature in self.expected_features:
      if feature in features.columns:
        aligned_features[feature] = features[feature]
      else:
        aligned_features[feature] = 0
    
    return aligned_features[self.expected_features]
  
  def predict_rul(self, df_processed: pd.DataFrame) -> float:
    if not self.model_loaded or self.model is None:
      return self._estimate_rul_heuristic(df_processed)[0]
    
    try:
      features_for_model = self._prepare_for_prediction(df_processed)
      if len(features_for_model) == 0:
        return self._estimate_rul_heuristic(df_processed)[0]
      
      if hasattr(self.model, 'predict'):
        predictions = self.model.predict(features_for_model)
      else:
        predictions = np.array([100])
      
      if hasattr(predictions, 'flatten'):
        predictions = predictions.flatten()
      
      return float(np.mean(predictions))
      
    except Exception as e:
      print(f"Error in model prediction: {e}")
      return self._estimate_rul_heuristic(df_processed)[0]
  
  def _prepare_for_prediction(self, df_processed: pd.DataFrame) -> pd.DataFrame:
    try:
      features = df_processed.copy()
      columns_to_drop = ['unit', 'RUL', 'cycle', 'max_cycle']
      for col in columns_to_drop:
        if col in features.columns:
          features = features.drop(columns=[col])
      
      features = features.fillna(0)
      
      for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
          features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
      
      if self.expected_features is not None:
        features = self._align_features_with_model(features)
      
      return features
      
    except Exception as e:
      print(f"Error preparing features: {e}")
      return pd.DataFrame()
  
  def _estimate_rul_heuristic(self, df_processed: pd.DataFrame) -> Tuple[float, str, str, float, float]:
    if len(df_processed) == 0:
      return 100.0, "LOW", COLORS["success"], 0.0, 0.0
    
    try:
      if 'cycle' in df_processed.columns:
        min_cycle = df_processed['cycle'].min() if len(df_processed) > 0 else 1
        if min_cycle <= 5:
          return 180.0, "LOW", COLORS["success"], 0.0, 0.0
      
      thermal_stress_mean = 0.0
      if "thermal_stress" in df_processed.columns:
        thermal_stress_mean = float(df_processed["thermal_stress"].mean())
      
      pressure_ratio_mean = 1.0
      if "pressure_ratio" in df_processed.columns:
        pressure_ratio_mean = float(df_processed["pressure_ratio"].mean())
      
      risk_score = 0
      if thermal_stress_mean > 100:
        risk_score += 2
      elif thermal_stress_mean > 50:
        risk_score += 1
      
      if pressure_ratio_mean > 3.0:
        risk_score += 2
      elif pressure_ratio_mean > 2.0:
        risk_score += 1
      
      if risk_score == 0:
        rul_estimate = float(np.random.randint(180, 250))
        risk_level = "LOW"
        risk_color = COLORS["success"]
      elif risk_score <= 2:
        rul_estimate = float(np.random.randint(100, 180))
        risk_level = "MODERATE"
        risk_color = COLORS["warning"]
      else:
        rul_estimate = float(np.random.randint(30, 100))
        risk_level = "HIGH"
        risk_color = COLORS["danger"]
      
      return rul_estimate, risk_level, risk_color, thermal_stress_mean, pressure_ratio_mean
      
    except Exception as e:
      print(f"Heuristic estimation error: {e}")
      return 150.0, "LOW", COLORS["success"], 0.0, 0.0
  
  def assess_engine_health(self, df_processed: pd.DataFrame, unit_id: Optional[int] = None) -> Tuple[float, str, str, float, float]:
    try:
      if unit_id is not None:
        df_unit = df_processed[df_processed["unit"] == unit_id].copy()
      else:
        df_unit = df_processed.copy()
      
      if len(df_unit) == 0:
        return 150.0, "LOW", COLORS["success"], 0.0, 0.0
      
      if self.model_loaded and self.model is not None:
        try:
          rul_estimate = self.predict_rul(df_unit)
          
          if rul_estimate > 180:
            risk_level = "LOW"
            risk_color = COLORS["success"]
          elif rul_estimate > 100:
            risk_level = "MODERATE"
            risk_color = COLORS["warning"]
          else:
            risk_level = "HIGH"
            risk_color = COLORS["danger"]
          
          thermal_stress = 0.0
          if "thermal_stress" in df_unit.columns:
            thermal_stress = float(df_unit["thermal_stress"].mean())
          
          pressure_ratio = 1.0
          if "pressure_ratio" in df_unit.columns:
            pressure_ratio = float(df_unit["pressure_ratio"].mean())
          
          return rul_estimate, risk_level, risk_color, thermal_stress, pressure_ratio
          
        except Exception as model_error:
          print(f"Model prediction failed, using heuristic: {model_error}")
          return self._estimate_rul_heuristic(df_unit)
      else:
        return self._estimate_rul_heuristic(df_unit)
        
    except Exception as e:
      print(f"Engine health assessment error: {e}")
      return 150.0, "LOW", COLORS["success"], 0.0, 0.0
  
  def get_sensor_columns(self, df: pd.DataFrame) -> list:
    return [col for col in df.columns if "sensor_measure" in col and col in df.columns]
  
  def get_sensor_data(self, df: pd.DataFrame, sensor_id: int, unit_id: Optional[int] = None) -> Optional[pd.Series]:
    sensor_col = f"sensor_measure_{sensor_id}"
    if sensor_col not in df.columns:
      return None
    
    if unit_id is not None:
      df_unit = df[df["unit"] == unit_id]
    else:
      df_unit = df
    
    return df_unit[sensor_col]

engine_analyzer = EngineAnalyzer()