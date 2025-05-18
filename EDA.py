import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'

# 데이터 경로 설정
BASE_PATH = Path('datasets/train')

# 각 카테고리별 데이터 로드 함수
def load_category_data(category_path):
    files = list(category_path.glob('*.parquet'))
    dfs = []
    for file in files:
        df = pd.read_parquet(file)
        dfs.append(df)
    return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()

# 각 카테고리별 데이터 로드
categories = {
    '회원정보': load_category_data(BASE_PATH / '1.회원정보'),
    '신용정보': load_category_data(BASE_PATH / '2.신용정보'),
    '승인매출정보': load_category_data(BASE_PATH / '3.승인매출정보'),
    '청구입금정보': load_category_data(BASE_PATH / '4.청구입금정보'),
    '잔액정보': load_category_data(BASE_PATH / '5.잔액정보'),
    '채널정보': load_category_data(BASE_PATH / '6.채널정보'),
    '마케팅정보': load_category_data(BASE_PATH / '7.마케팅정보'),
    '성과정보': load_category_data(BASE_PATH / '8.성과정보')
}

# 각 카테고리별 기본 정보 출력
for category, df in categories.items():
    print(f"\n=== {category} 데이터 정보 ===")
    print(f"데이터 크기: {df.shape}")
    print("\n컬럼 목록:")
    print(df.columns.tolist())
    print("\n결측치 정보:")
    print(df.isnull().sum())
    print("\n기본 통계:")
    print(df.describe())

# 시각화 함수
def plot_category_distribution(df, category_name):
    plt.figure(figsize=(15, 10))
    
    # 수치형 변수들의 분포
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(numeric_cols[:4], 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df, x=col, bins=30)
        plt.title(f'{category_name} - {col} 분포')
    
    plt.tight_layout()
    plt.savefig(f'{category_name}_distribution.png')
    plt.close()

# 각 카테고리별 분포 시각화
for category, df in categories.items():
    plot_category_distribution(df, category)

# 상관관계 분석
def plot_correlation_matrix(df, category_name):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'{category_name} 상관관계')
    plt.tight_layout()
    plt.savefig(f'{category_name}_correlation.png')
    plt.close()

# 각 카테고리별 상관관계 시각화
for category, df in categories.items():
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plot_correlation_matrix(numeric_df, category) 