import pandas as pd
from fbprophet import Prophet

class MMTimeSeriesAnalysis:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.model = None
        self.forecast = None
        
    def train(self):
        self.model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        self.model.fit(self.data)
        
    def predict(self, days=30):
        future = self.model.make_future_dataframe(periods=days)
        self.forecast = self.model.predict(future)
