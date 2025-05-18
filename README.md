# Dacon 신용카드 사용자 연체 예측 대회

이 프로젝트는 Dacon에서 주최한 신용카드 사용자 연체 예측 대회를 위한 코드입니다. 다양한 머신러닝 모델과 딥러닝 모델을 사용하여 신용카드 사용자의 연체 여부를 예측합니다.

## 프로젝트 구조

### 1. 데이터 분석 및 전처리
- `EDA.py`: 탐색적 데이터 분석
- `main.ipynb`: 메인 분석 노트북
- `main_for_small.ipynb`: 소규모 데이터 분석

### 2. 모델 학습
- `deep-train.py`: 딥러닝 모델 학습
- `deep-learnign-full.ipynb`: 전체 딥러닝 실험
- `main_dl.ipynb`: 딥러닝 모델 구현

### 3. 결과 및 제출
- `deep-submission.csv`: 딥러닝 모델 제출 결과
- `random_forest_submit.csv`: 랜덤 포레스트 제출 결과
- `xgboost_submit.csv`: XGBoost 제출 결과

### 4. 시각화
- `*_correlation.png`: 각 특성별 상관관계 시각화
- `*_distribution.png`: 각 특성별 분포 시각화

## 필요 조건

- Python 3.8 이상
- PyTorch
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn

## 설치 방법

1. 저장소를 클론합니다:
```bash
git clone [repository-url]
```

2. 필요한 패키지를 설치합니다:
```bash
pip install torch scikit-learn xgboost pandas numpy matplotlib seaborn
```

## 사용 방법

### 데이터 분석
```bash
python EDA.py
```

### 모델 학습
```bash
python deep-train.py
```

### Jupyter Notebook 실행
```bash
jupyter notebook
```

## 주요 기능

1. **데이터 전처리**
   - 결측치 처리
   - 이상치 제거
   - 특성 엔지니어링
   - 데이터 정규화

2. **모델 구현**
   - 딥러닝 모델 (PyTorch)
   - 랜덤 포레스트
   - XGBoost
   - 앙상블 방법

3. **특성 분석**
   - 상관관계 분석
   - 분포 분석
   - 중요도 분석
   - 시각화

4. **모델 평가**
   - 교차 검증
   - 성능 메트릭 계산
   - 예측 결과 분석

## 데이터셋 구조

1. **회원정보**
   - 기본 인적사항
   - 가입 정보
   - 회원 등급

2. **신용정보**
   - 신용 점수
   - 대출 정보
   - 연체 이력

3. **거래정보**
   - 승인 매출
   - 청구 입금
   - 잔액 정보

4. **마케팅정보**
   - 마케팅 채널
   - 프로모션
   - 구매 패턴

## 참고 자료

- [Dacon 대회 페이지](https://dacon.io/competitions/official/235713/overview/)
- [PyTorch 문서](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn 문서](https://scikit-learn.org/stable/)
- [XGBoost 문서](https://xgboost.readthedocs.io/) 