import traceback
import pandas as pd
from joblib import load
from types import SimpleNamespace
from flask import Flask, request, jsonify
from moduleDatabase import DatabaseMethods


app = Flask(__name__)
if not app:
    app = Flask(__name__)


@app.route("/")
def hello():
    return "Welcome to MCI Detection API! You are currently at root."


@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)
            sample = SimpleNamespace()
            sample.education = json_['education']
            sample.laptop_usage = json_['laptop_usage']
            sample.smartphone_usage = json_['smartphone_usage']
            sample.family_med_history = json_['family_med_history']
            sample.exercising = json_['exercising']
            # sample.marital_status = json_['marital_status']
            sample.hypertension = json_['hypertension']
            sample.total_win_gr_points_in_gs = json_['total_win_gr_points_in_gs']
            sample.anaklisiImp = json_['anaklisiImp']
            sample.orientImp = json_['orientImp']
            sample.logicImp = json_['logicImp']
            # predicted_class = model.predict([[sample.education, sample.p2, sample.p3, sample.p4]])
            test = pd.DataFrame({
                'education': [sample.education],
                'laptop_usage': [sample.laptop_usage],
                'smartphone_usage': [sample.smartphone_usage],
                'family_med_history': [sample.family_med_history],
                'exercising': [sample.exercising],
                # 'marital_status': [sample.marital_status],
                'hypertension': [sample.hypertension],
                'total_win_gr_points_in_gs': [sample.total_win_gr_points_in_gs],
                'anaklisiImp': [sample.anaklisiImp],
                'orientImp': [sample.orientImp],
                'logicImp': [sample.logicImp]
            })
            result = model.predict(test)

            # select
            # label
            # from target_labels tl
            # where
            # tl.bin = 1 and tl.target_class = 'moca_pre_binary_binned';
            db = DatabaseMethods()
            sql = 'select label from target_labels tl where tl.bin=%d and tl.target_class=\'moca_pre_binary_binned\''
            df = db.fetch(sql % (result))
            print(df['label'].iloc[0])

            print('PREDICTION RAW: ', result, 'Label:', df['label'].iloc[0])
            # return jsonify({'prediction': str(result)})
            return jsonify({'Prediction raw': str(result), 'Label:': df['label'].iloc[0]})
        except Exception as e:
            print(e)
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':
    port = 5000

    pathFileName = 'C:\\Users\\christos\\thesisProject\\thesis\\models\\model.joblib'
    with open(pathFileName, 'rb') as f:
        model = load(f)
        print(type(model))
    print('Model loaded')
    # model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"auto_init
    print('Model columns loaded')

    app.run(port=port, debug=False)
