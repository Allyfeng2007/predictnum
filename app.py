from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# 加载模型
model = joblib.load('linreg.model')

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    data = request.get_json()
    
    # 验证输入
    if not data or 'features' not in data:
        return jsonify({'error': '缺少features参数'}), 400
    
    features = data['features']
    
    if len(features) != 3:
        return jsonify({'error': '需要3个特征值'}), 400
    
    if not all(isinstance(x, (int, float)) for x in features):
        return jsonify({'error': '特征值必须为数字'}), 400

    # 转换为模型需要的格式并预测
    features_array = np.array(features).reshape(1, -1)
    df = pd.DataFrame(features_array, columns=['TV', 'Weibo', 'WeChat'])
    prediction = model.predict(df)[0]
    
    # 返回结果
    return jsonify({
        'success': True,
        'prediction': float(prediction),
        'features': features
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)