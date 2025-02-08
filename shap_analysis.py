import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data():
    """加载训练数据"""
    # 加载匹配后的数据
    df = pd.read_csv('/Volumes/yasen/cursor/matched_data.csv')
    
    # 定义特征列（与app.py中保持一致）
    feature_columns = [
        'Age', 'Sex', 'Marital_status', 'Residence', 'Smoking', 'Drinking',
        'Physical_disabilities', 'Mental_retardation', 'Vision_problem',
        'Hearing_problem', 'Speech_impediment', 'difficulty_with_getting_up',
        'difficulty_with_stooping'
    ] + [f'new_da007_{i}_' for i in range(1, 15)]
    
    # 特征的英文名称映射
    feature_names_en = {
        'Age': 'Age',
        'Sex': 'Sex',
        'Marital_status': 'Marital Status',
        'Residence': 'Residence',
        'Smoking': 'Smoking',
        'Drinking': 'Drinking',
        'Physical_disabilities': 'Physical Disabilities',
        'Mental_retardation': 'Mental Retardation',
        'Vision_problem': 'Vision Problem',
        'Hearing_problem': 'Hearing Problem',
        'Speech_impediment': 'Speech Impediment',
        'difficulty_with_getting_up': 'Difficulty with Getting Up',
        'difficulty_with_stooping': 'Difficulty with Stooping',
        'new_da007_1_': 'Hypertension',
        'new_da007_2_': 'Dyslipidemia',
        'new_da007_3_': 'Diabetes',
        'new_da007_4_': 'Cancer',
        'new_da007_5_': 'Chronic Lung Disease',
        'new_da007_6_': 'Liver Disease',
        'new_da007_7_': 'Heart Disease',
        'new_da007_8_': 'Stroke',
        'new_da007_9_': 'Kidney Disease',
        'new_da007_10_': 'Digestive Disease',
        'new_da007_11_': 'Psychiatric Problems',
        'new_da007_12_': 'Memory-related Disease',
        'new_da007_13_': 'Arthritis',
        'new_da007_14_': 'Asthma'
    }
    
    # 准备特征数据
    X = df[feature_columns]
    
    # 分别获取2018和2020年的目标变量
    y_2018 = df['fall_2018']  # 2018年跌倒状态（0=未发生，1=发生）
    y_2020 = df['fall_2020']  # 2020年跌倒状态（0=未发生，1=发生）
    
    # 使用英文特征名
    feature_names = [feature_names_en[col] for col in feature_columns]
    
    # 添加数据验证
    print("数据基本信息：")
    print(f"总样本数：{len(df)}")
    print("\n2018年跌倒情况：")
    print(y_2018.value_counts())
    print("\n2020年跌倒情况：")
    print(y_2020.value_counts())
    
    # 检查缺失值
    missing = X.isnull().sum()
    if missing.any():
        print("\n特征缺失值情况：")
        print(missing[missing > 0])
    
    return X, y_2018, X, y_2020, feature_names

def load_models():
    """加载训练好的模型"""
    model_2018 = joblib.load('/Volumes/yasen/cursor/models/model_fall_2018.pkl')
    model_2020 = joblib.load('/Volumes/yasen/cursor/models/model_fall_2020.pkl')
    return model_2018, model_2020

def generate_shap_plots(model, X, feature_names, model_name, output_dir='/Volumes/yasen/cursor/shap_plots'):
    """生成SHAP值分析图"""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n生成 {model_name} 的SHAP值分析...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统可用的中文字体
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 打印SHAP值的形状以进行调试
    print("原始SHAP值形状:", np.array(shap_values).shape)
    
    try:
        # 我们只关注正类（发生跌倒）的SHAP值
        if isinstance(shap_values, list) and len(shap_values) == 2:
            print("使用正类(跌倒)的SHAP值")
            shap_values = shap_values[1]  # 取正类的SHAP值
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            print("从3D数组中提取正类SHAP值")
            shap_values = shap_values[:, :, 1]  # 取第二个类别的SHAP值
        
        print("处理后SHAP值形状:", shap_values.shape)
        
        # 将特征名称转换为列表
        feature_names = list(feature_names)
        
        # 1. 特征重要性汇总图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
            max_display=20
        )
        plt.title(f'{model_name.replace("跌倒风险", "Fall Risk")} - Feature Importance Summary')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_summary_plot.tiff', dpi=300, bbox_inches='tight', format='tiff')
        plt.close()
        
        # 2. 计算特征重要性并创建DataFrame
        importance_values = np.abs(shap_values).mean(axis=0)
        importance_dict = {
            'feature': feature_names,
            'importance': importance_values
        }
        feature_importance = pd.DataFrame(importance_dict)
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        # 绘制特征重要性条形图
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Feature Importance (Mean |SHAP value|)')
        plt.title(f'{model_name.replace("跌倒风险", "Fall Risk")} - Feature Importance Ranking')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_feature_importance.tiff', dpi=300, bbox_inches='tight', format='tiff')
        plt.close()
        
        # 3. 为每个特征生成依赖图
        for i, feature_name in enumerate(feature_names):
            plt.figure(figsize=(10, 6))
            feature_col = X.columns[i]
            
            # 使用scatter plot替代dependence plot
            plt.scatter(X[feature_col], shap_values[:, i], alpha=0.5)
            plt.xlabel(feature_name)
            plt.ylabel('SHAP Value')
            plt.title(f'{model_name.replace("跌倒风险", "Fall Risk")} - {feature_name} Dependence Plot')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{model_name}_{feature_name}_dependence.tiff', dpi=300, bbox_inches='tight', format='tiff')
            plt.close()
        
        return feature_importance
        
    except Exception as e:
        print(f"生成图表时出错: {str(e)}")
        print("SHAP值形状:", np.array(shap_values).shape if isinstance(shap_values, list) else shap_values.shape)
        print("特征数量:", len(feature_names))
        print("数据形状:", X.shape)
        print("特征名称:", feature_names)
        if isinstance(shap_values, list):
            print("SHAP值列表长度:", len(shap_values))
            for i, sv in enumerate(shap_values):
                print(f"第{i}个SHAP值数组形状:", sv.shape)
        raise e

def create_feature_importance_comparison(importance_2018, importance_2020, output_dir='/Volumes/yasen/cursor/shap_plots'):
    """创建2018和2020模型特征重要性对比图"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统可用的中文字体
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(15, 10))
    
    # 合并两个模型的特征重要性
    comparison_df = pd.merge(
        importance_2018.rename(columns={'importance': '2018'}),
        importance_2020.rename(columns={'importance': '2020'}),
        on='feature'
    )
    
    # 计算平均重要性并排序
    comparison_df['avg_importance'] = (comparison_df['2018'] + comparison_df['2020']) / 2
    comparison_df = comparison_df.sort_values('avg_importance', ascending=True)
    
    # 创建对比条形图
    y_pos = np.arange(len(comparison_df))
    width = 0.35
    
    plt.barh(y_pos - width/2, comparison_df['2018'], width, label='2018 Model', color='#2ecc71', alpha=0.8)
    plt.barh(y_pos + width/2, comparison_df['2020'], width, label='2020 Model', color='#3498db', alpha=0.8)
    
    plt.yticks(y_pos, comparison_df['feature'])
    plt.xlabel('Feature Importance (Mean |SHAP value|)')
    plt.title('2018 vs 2020 Model Feature Importance Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_comparison.tiff', dpi=300, bbox_inches='tight', format='tiff')
    plt.close()

def main():
    # 1. 加载数据
    X, y_2018, X, y_2020, feature_names = load_data()
    
    # 2. 加载模型
    model_2018, model_2020 = load_models()
    
    output_dir = '/Volumes/yasen/cursor/shap_plots'
    
    # 3. 生成SHAP分析图
    importance_2018 = generate_shap_plots(model_2018, X, feature_names, '2018 Model (Fall Risk)', output_dir)
    importance_2020 = generate_shap_plots(model_2020, X, feature_names, '2020 Model (Fall Risk)', output_dir)
    
    # 4. 创建特征重要性对比图
    create_feature_importance_comparison(importance_2018, importance_2020, output_dir)
    
    # 5. 打印特征重要性排序
    print("\n2018模型跌倒风险特征重要性排序：")
    print(importance_2018)
    print("\n2020模型跌倒风险特征重要性排序：")
    print(importance_2020)
    
    # 6. 保存特征重要性到CSV文件
    importance_2018.to_csv(f'{output_dir}/feature_importance_2018.csv', index=False, encoding='utf-8-sig')
    importance_2020.to_csv(f'{output_dir}/feature_importance_2020.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main() 