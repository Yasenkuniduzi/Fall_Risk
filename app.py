from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import numpy as np
import logging
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于会话管理

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义特征列表
feature_columns = [
    'Age', 'Sex', 'Marital_status', 'Residence', 'Smoking', 'Drinking',
    'Physical_disabilities', 'Mental_retardation', 'Vision_problem',
    'Hearing_problem', 'Speech_impediment', 'difficulty_with_getting_up',
    'difficulty_with_stooping'
] + [f'new_da007_{i}_' for i in range(1, 15)]

# 加载模型
model_2018 = joblib.load('models/model_fall_2018.pkl')
model_2020 = joblib.load('models/model_fall_2020.pkl')

# 修改静态文件配置
app.config['FREEZER_BASE_URL'] = 'https://yasenkuniduzi.github.io/cursor/'
app.config['FREEZER_RELATIVE_URLS'] = True

def validate_input(data):
    """验证输入数据"""
    errors = []
    try:
        # 检查是否所有必需字段都存在
        required_fields = feature_columns + ['name']
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        if missing_fields:
            errors.append(f"以下字段为必填项: {', '.join(missing_fields)}")
            return errors

        age = float(data.get('Age', 0))
        if not (40 <= age <= 120):
            errors.append("年龄必须在40-120岁之间")
        
        binary_fields = [
            'Sex', 'Marital_status', 'Physical_disabilities', 'Mental_retardation',
            'Vision_problem', 'Hearing_problem', 'Speech_impediment',
            'Smoking', 'Drinking', 'difficulty_with_getting_up',
            'difficulty_with_stooping'
        ] + [f'new_da007_{i}_' for i in range(1, 15)]
        
        for field in binary_fields:
            value = data.get(field)
            if not value or value not in ['1', '2']:
                errors.append(f"'{field}' 必须选择 '是' 或 '否'")
    
    except ValueError as e:
        logger.error(f"输入数据验证错误: {str(e)}")
        errors.append(f"输入数据格式错误: {str(e)}")
    except Exception as e:
        logger.error(f"验证过程中发生未知错误: {str(e)}")
        errors.append("验证过程中发生错误，请检查输入数据")
    
    return errors

def generate_recommendations(data):
    """根据患者个体情况生成个性化预防建议"""
    recommendations = []
    
    # 基于年龄的建议
    age = int(data.get('Age', 0))
    if age >= 80:
        recommendations.append("建议每天进行适度的平衡训练，如太极拳或站立练习 / Recommend daily balance training such as Tai Chi or standing practice")
    
    # 基于身体状况的建议
    if data.get('Vision_problem') == '1':
        recommendations.append("建议定期进行视力检查，保持眼镜度数适中 / Regular vision check-ups and maintain appropriate glasses prescription")
    
    if data.get('Hearing_problem') == '1':
        recommendations.append("建议佩戴助听器，保持声音环境清晰 / Consider using hearing aids and maintain a clear sound environment")
    
    if data.get('difficulty_with_getting_up') == '1':
        recommendations.append("建议在床边和浴室安装扶手，使用防滑垫 / Install handrails near bed and bathroom, use non-slip mats")
    
    if data.get('difficulty_with_stooping') == '1':
        recommendations.append("建议使用取物夹或加高便座等辅助工具 / Use reaching aids or raised toilet seats")
    
    # 基于慢性病的建议
    if data.get('new_da007_1_') == '1':  # 高血压
        recommendations.append("建议定期监测血压，按时服药，避免剧烈运动 / Regular blood pressure monitoring, take medicine on time, avoid intense exercise")
    
    if data.get('new_da007_7_') == '1':  # 心脏病
        recommendations.append("建议进行低强度有氧运动，避免过度疲劳 / Recommend low-intensity aerobic exercise, avoid over-fatigue")
    
    if data.get('new_da007_13_') == '1':  # 关节炎
        recommendations.append("建议进行水中运动或其他低冲击性运动 / Consider water exercises or other low-impact activities")
    
    # 生活习惯相关建议
    if data.get('Smoking') == '1':
        recommendations.append("建议戒烟，可寻求戒烟门诊帮助 / Consider quitting smoking, seek professional help if needed")
    
    if data.get('Drinking') == '1':
        recommendations.append("建议限制饮酒，避免醉酒 / Limit alcohol consumption, avoid intoxication")
    
    # 环境相关建议
    if data.get('Residence') == '1':  # 城市
        recommendations.append("建议保持室内光线充足，清理地面杂物 / Maintain good indoor lighting and clear floor obstacles")
    else:  # 农村
        recommendations.append("建议改善居住环境，铺设防滑地板，保持道路平整 / Improve living environment, install non-slip flooring, maintain even pathways")
    
    # 确保建议不超过10条
    if len(recommendations) > 10:
        recommendations = recommendations[:10]
    
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        logger.info(f"收到预测请求: {data}")
        
        if not data:
            logger.error("没有收到表单数据")
            return render_template('index.html', 
                                error="请填写表单后再提交 / Please fill the form before submitting")
        
        # 输入验证
        errors = validate_input(data)
        if errors:
            logger.warning(f"输入验证失败: {errors}")
            return render_template('index.html', 
                                errors=errors,
                                form_data=data)
        
        try:
            features = []
            for col in feature_columns:
                val = data.get(col, '')
                if not val:
                    raise ValueError(f"缺少必要的特征值: {col}")
                features.append(float(val))
            
            features = np.array(features).reshape(1, -1)
            
            # 确保模型加载成功
            if model_2018 is None or model_2020 is None:
                raise Exception("模型未正确加载")
            
            risk_2018 = float(model_2018.predict_proba(features)[0][1])
            risk_2020 = float(model_2020.predict_proba(features)[0][1])
            
            recommendations = generate_recommendations(data)
            
            session['report_data'] = {
                'name': data.get('name', '未知'),
                'age': data.get('Age', '未知'),
                'sex': '男 / Male' if data.get('Sex') == '1' else '女 / Female',
                'marital_status': '已婚 / Married' if data.get('Marital_status') == '1' else '其他 / Other',
                'residence': '城市 / Urban' if data.get('Residence') == '1' else '农村 / Rural',
                'risk_2018': f"{risk_2018:.2%}",
                'risk_2020': f"{risk_2020:.2%}",
                'recommendations': recommendations,
                'report_date': datetime.now().strftime('%Y年%m月%d日 %H:%M')
            }
            
            return redirect(url_for('report'))
            
        except ValueError as e:
            logger.error(f"数据处理错误: {str(e)}")
            return render_template('index.html', 
                                error="数据处理错误，请确保所有输入都是有效的",
                                form_data=data)
    except Exception as e:
        error_msg = f"预测过程中发生错误: {str(e)}"
        logger.error(error_msg)
        return render_template('index.html', 
                             error="服务器处理请求时发生错误，请稍后重试",
                             form_data=data)

@app.route('/report')
def report():
    data = session.get('report_data', {})
    return render_template('report.html', **data)

# 修改运行配置，确保在 GitHub Pages 上正常工作
if __name__ == '__main__':
    # 本地开发时使用
    app.run(debug=True, host='0.0.0.0', port=5000) 