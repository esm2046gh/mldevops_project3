from ml.data import Dump
from ml.model import model_performance
from ml.model import model_performance_on_slices

if __name__ == "__main__":
    dump = Dump('joblib')
    trained_models = {"dtc", "lrc", "rfc", "svc"}
    for model_key in trained_models:
       model = dump.load(f'../model/{model_key}/model.pkl')
       train = dump.load(f'../model/{model_key}/train.pkl')
       test = dump.load(f'../model/{model_key}/test.pkl')
       cat_features = dump.load(f'../model/{model_key}/cat_features.pkl')
       output_feature = dump.load(f'../model/{model_key}/output_feature.pkl')
       encoder = dump.load(f'../model/{model_key}/encoder.pkl')
       lb = dump.load(f'../model/{model_key}/lb.pkl')
       scaler = dump.load(f'../model/{model_key}/scaler.pkl')

       print(f"Overall Performance({model_key}): Test")
       model_performance(model, test, cat_features, output_feature, encoder, lb, scaler)
       #print(f"Slice Performance({model_key}): Test")
       #model_performance_on_slices(model, test, cat_features, output_feature, encoder, lb, scaler)


