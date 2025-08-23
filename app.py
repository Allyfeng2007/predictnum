from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# 全局变量
model = None
model_loaded = False  # 添加一个标志位


def load_model():
    """加载模型"""
    global model, model_loaded
    try:
        model = joblib.load('linreg.model')
        model_loaded = True
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise


@app.before_request
def before_first_request():
    """在第一个请求前加载模型"""
    global model_loaded
    if not model_loaded:
        load_model()


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'healthy', 'service': 'ml-prediction'})


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        # 检查模型是否已加载
        if not model_loaded or model is None:
            return jsonify({'error': '模型未加载，请稍后重试'}), 503

        # 获取请求数据
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features field'}), 400

        features = data['features']

        # 验证输入
        if len(features) != 3:
            return jsonify({'error': '需要3个特征值'}), 400

        if not all(isinstance(x, (int, float)) for x in features):
            return jsonify({'error': '特征值必须为数字'}), 400

        # 预测
        features_array = np.array(features).reshape(1, -1)
        df = pd.DataFrame(features_array, columns=['TV', 'Weibo', 'WeChat'])
        prediction = model.predict(df)[0]

        # 返回结果
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'features': features
        }),

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 启动时直接加载模型，避免第一个请求的延迟
    load_model()
    app.run(host='0.0.0.0', port=80, debug=False)
