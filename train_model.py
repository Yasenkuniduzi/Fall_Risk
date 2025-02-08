import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import warnings
import os
from datetime import datetime

# 忽略警告
warnings.filterwarnings('ignore')

# 设置日志
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建必要的目录
for directory in ['models', 'plots', 'logs']:
    os.makedirs(directory, exist_ok=True)

def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
            
        # 读取数据
        df = pd.read_csv(file_path)
        logger.info(f"成功加载数据，共 {len(df)} 条记录")
        
        # 检查数据集中的实际列名
        logger.info("数据集包含的列名：")
        logger.info("\n".join(df.columns.tolist()))
        
        # 定义特征列（根据实际数据集的列名）
        feature_columns = [
            'Age', 'Sex', 'Marital_status', 'Residence', 'Smoking', 'Drinking',
            'Physical_disabilities', 'Mental_retardation', 'Vision_problem',
            'Hearing_problem', 'Speech_impediment', 'difficulty_with_getting_up',
            'difficulty_with_stooping'
        ] + [f'new_da007_{i}_' for i in range(1, 15)]
        
        # 验证所有特征列是否存在
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"以下列在数据集中不存在: {missing_columns}")
            raise KeyError(f"缺少必要的列: {missing_columns}")
        
        # 检查并处理缺失值
        missing_values = df[feature_columns + ['fall_2018', 'fall_2020']].isnull().sum()
        if missing_values.any():
            logger.warning(f"发现缺失值:\n{missing_values[missing_values > 0]}")
            
            # 使用众数填充分类变量的缺失值
            categorical_cols = ['Sex', 'Marital_status', 'Residence', 'Smoking', 'Drinking'] + \
                             [f'new_da007_{i}_' for i in range(1, 15)]
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
            
            # 使用中位数填充连续变量的缺失值
            df['Age'].fillna(df['Age'].median(), inplace=True)
        
        # 数据类型转换
        df['Age'] = df['Age'].astype(float)
        categorical_columns = [col for col in feature_columns if col != 'Age']
        df[categorical_columns] = df[categorical_columns].astype('category')
        
        return df, feature_columns
    
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise

def train_and_evaluate_model(X_train, X_test, y_train, y_test, target_name):
    """训练和评估模型"""
    try:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # 模型评估
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        # 打印评估报告
        logger.info(f"\n{target_name} 模型评估报告：\n{classification_report(y_test, y_pred)}")
        
        return rf, y_test, y_pred_proba
    
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        raise

def plot_roc_curve(y_test, y_pred_proba, target_name):
    """绘制ROC曲线"""
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title(f'{target_name} ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig(f'plots/roc_curve_{target_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    except Exception as e:
        logger.error(f"ROC曲线绘制失败: {str(e)}")
        raise

def plot_feature_importance(rf, feature_columns, target_name):
    """绘制特征重要性图"""
    try:
        plt.figure(figsize=(12, 8))
        importances = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=importances.head(15), x='importance', y='feature')
        plt.title(f'{target_name} 特征重要性 Top 15')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.savefig(f'plots/feature_importance_{target_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    except Exception as e:
        logger.error(f"特征重要性图绘制失败: {str(e)}")
        raise

def main():
    try:
        # 加载数据
        df, feature_columns = load_and_preprocess_data('matched_data.csv')
        
        # 检查数据基本情况
        logger.info("\n数据集基本信息:")
        logger.info(f"数据集形状: {df.shape}")
        logger.info("\n特征列数据类型:")
        logger.info(df[feature_columns].dtypes)
        
        # 训练模型
        for target in ['fall_2018', 'fall_2020']:
            if target not in df.columns:
                logger.error(f"目标变量 {target} 不在数据集中")
                continue
            
            logger.info(f"\n开始训练 {target} 模型")
            logger.info(f"目标变量分布:\n{df[target].value_counts(normalize=True)}")
            
            X = df[feature_columns]
            y = df[target]
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 训练和评估模型
            rf, y_test, y_pred_proba = train_and_evaluate_model(
                X_train, X_test, y_train, y_test, target
            )
            
            # 绘制ROC曲线
            plot_roc_curve(y_test, y_pred_proba, target)
            
            # 绘制特征重要性
            plot_feature_importance(rf, feature_columns, target)
            
            # 保存模型
            model_path = f'models/model_{target}.pkl'
            joblib.dump(rf, model_path)
            logger.info(f"{target} 模型已保存到 {model_path}")
        
        logger.info("所有模型训练完成")
        
    except Exception as e:
        logger.error(f"训练过程发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main() 