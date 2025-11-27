def load_nasa_data(train, test, true_rul):
  column_names= ["unit", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_measure_{i}" for i in range(1, 22)]
  train_data= pd.read_csv(train, sep=r"\s+", header=None, names=column_names)
  test_data=  pd.read_csv(test, sep=r"\s+", header=None, names=column_names)
  true_rul=   pd.read_csv(true_rul, sep=r"\s+", header=None, names=["true_rul"])

  return train_data, test_data, true_rul, column_names