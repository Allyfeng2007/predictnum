from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# 加载模型
model = joblib.load('linreg.model')

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口（建议保留）"""
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or 'features' not in data:
        return jsonify({'error': '缺少features参数'}), 400
    
    features = data['features']
    
    if len(features) != 3:
        return jsonify({'error': '需要3个特征值'}), 400
    
    features_array = np.array(features).reshape(1, -1)
    df = pd.DataFrame(features_array, columns=['TV', 'Weibo', 'WeChat'])
    prediction = model.predict(df)[0]
    
    return jsonify({
        'success': True,
        'prediction': float(prediction),
        'features': features
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)
