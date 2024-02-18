
from src.InsurancePremiumPrediction.pipelines.prediction_pipeline import CustomData

custobj = CustomData(64,'male',24.7,1,'no','northwest')

data = custobj.get_data_as_dataframe()

print(data)