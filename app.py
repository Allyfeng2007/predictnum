from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# 全局模型变量
model = None

def load_model():
    """加载模型"""
    global model
    try:
        model = joblib.load('linreg.model')
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'healthy', 'service': 'ml-prediction'})

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
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
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 启动时加载模型
@app.before_first_request
def initialize():
    load_model()

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=80, debug=False)