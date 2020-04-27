import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#根据特征重要性绘制图
def plot_feature_importances(df, n=15, color='blue'):
    # Sort features with most important at the head
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    # 归一化处理
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])#计算累计重要性

    plt.rcParams['font.size'] = 12
    plt.style.use('fivethirtyeight')
    # Bar plot of n most important features
    df.iloc[:n, :].plot.barh(y='importance_normalized',
                            x='feature', color=color,
                            edgecolor='k', figsize=(14, 8),
                            legend=False)

    plt.xlabel('Normalized Importance', size=18);
    plt.ylabel('');
    plt.title('Top %d Most Important Features' %n, size=18)
    plt.gca().invert_yaxis()#将实际中的y轴顺序反转
    plt.tight_layout()#紧密布局
    #plt.savefig('../input1/auto_importance.png')
    plt.savefig('../input1/auto_importance1.png')
    return df