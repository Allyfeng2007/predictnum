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

# before_request在每一次请求进入你的应用、匹配到路由之前都会先执行这个函数。典型用途：鉴权、打开数据库连接、懒加载模型、设置请求级别上下文等
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
    if not model_loaded or model is None:
        return jsonify({'error': '模型未加载，请稍后重试'}), 503

    data = request.get_json(silent=True)
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing features field'}), 400

    try:
        features = data['features']
        X = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]
        return jsonify({'prediction': float(pred)})

    except Exception as e:
        app.logger.exception(f"/predict 出错: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
