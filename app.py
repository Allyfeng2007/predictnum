from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = None
model_loaded = False

def load_model():
    global model, model_loaded
    try:
        model = joblib.load('linreg.model')
        model_loaded = True
        app.logger.info("模型加载成功")
    except Exception as e:
        app.logger.exception(f"模型加载失败: {e}")
        raise

# ✅ 用 before_first_request（更符合你的注释）
@app.before_request
def _load_once():
    global model_loaded
    if not model_loaded:
        load_model()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'ml-prediction'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model_loaded or model is None:
            return jsonify({'error': '模型未加载，请稍后重试'}), 503

        data = request.get_json(silent=True)
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features field'}), 400

        features = data['features']

        if len(features) != 3:
            return jsonify({'error': '需要3个特征值'}), 400
        if not all(isinstance(x, (int, float)) for x in features):
            return jsonify({'error': '特征值必须为数字'}), 400

        # ✅ 用 numpy，避免特征名问题
        X = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]

        # ✅ 去掉这个返回末尾的逗号！
        return jsonify({
            'success': True,
            'prediction': float(pred),
            'features': features
        })
    except Exception as e:
        app.logger.exception(f"/predict 出错: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
