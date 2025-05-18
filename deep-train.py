import gc
import time
import random
import pandas as pd
import numpy as np

# import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

# 커스텀 데이터셋 정의
class CardDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 데이터 분할(폴더) 구분
data_splits = ["train", "test"]

# 각 데이터 유형별 폴더명, 파일 접미사, 변수 접두어 설정
data_categories = {
    "회원정보": {"folder": "1.회원정보", "suffix": "회원정보", "var_prefix": "customer"},
    "신용정보": {"folder": "2.신용정보", "suffix": "신용정보", "var_prefix": "credit"},
    "승인매출정보": {"folder": "3.승인매출정보", "suffix": "승인매출정보", "var_prefix": "sales"},
    "청구정보": {"folder": "4.청구입금정보", "suffix": "청구정보", "var_prefix": "billing"},
    "잔액정보": {"folder": "5.잔액정보", "suffix": "잔액정보", "var_prefix": "balance"},
    "채널정보": {"folder": "6.채널정보", "suffix": "채널정보", "var_prefix": "channel"},
    "마케팅정보": {"folder": "7.마케팅정보", "suffix": "마케팅정보", "var_prefix": "marketing"},
    "성과정보": {"folder": "8.성과정보", "suffix": "성과정보", "var_prefix": "performance"}
}

# 2018년 7월부터 12월까지의 월 리스트
months = ['07', '08', '09', '10', '11', '12']

for split in data_splits:
    for category, info in data_categories.items():
        folder = info["folder"]
        suffix = info["suffix"]
        var_prefix = info["var_prefix"]
        
        for month in months:
            # 파일명 형식: 2018{month}_{split}_{suffix}.parquet
            file_path = f"./datasets/{split}/{folder}/2018{month}_{split}_{suffix}.parquet"
            # 변수명 형식: {var_prefix}_{split}_{month}
            variable_name = f"{var_prefix}_{split}_{month}"
            globals()[variable_name] = pd.read_parquet(file_path)
            print(f"{variable_name} is loaded from {file_path}")

gc.collect()

# 데이터 유형별 설정 
info_categories = ["customer", "credit", "sales", "billing", "balance", "channel", "marketing", "performance"]

# 월 설정
months = ['07', '08', '09', '10', '11', '12']

#### Train ####

# 각 유형별로 월별 데이터를 합쳐서 새로운 변수에 저장
train_dfs = {}

for prefix in info_categories:
    # globals()에서 동적 변수명으로 데이터프레임들을 가져와 리스트에 저장
    df_list = [globals()[f"{prefix}_train_{month}"] for month in months]
    train_dfs[f"{prefix}_train_df"] = pd.concat(df_list, axis=0)
    gc.collect()
    print(f"{prefix}_train_df is created with shape: {train_dfs[f'{prefix}_train_df'].shape}")
    # 원본 데이터프레임 삭제
    for month in months:
        del globals()[f"{prefix}_train_{month}"]
    gc.collect()

customer_train_df = train_dfs["customer_train_df"]
credit_train_df   = train_dfs["credit_train_df"]
sales_train_df    = train_dfs["sales_train_df"]
billing_train_df  = train_dfs["billing_train_df"]
balance_train_df  = train_dfs["balance_train_df"]
channel_train_df  = train_dfs["channel_train_df"]
marketing_train_df= train_dfs["marketing_train_df"]
performance_train_df = train_dfs["performance_train_df"]

del train_dfs
gc.collect()

#### Test ####

# test 데이터에 대해 train과 동일한 방법 적용
test_dfs = {}

for prefix in info_categories:
    df_list = [globals()[f"{prefix}_test_{month}"] for month in months]
    test_dfs[f"{prefix}_test_df"] = pd.concat(df_list, axis=0)
    gc.collect()
    print(f"{prefix}_test_df is created with shape: {test_dfs[f'{prefix}_test_df'].shape}")
    # 원본 데이터프레임 삭제
    for month in months:
        del globals()[f"{prefix}_test_{month}"]
    gc.collect()

customer_test_df = test_dfs["customer_test_df"]
credit_test_df   = test_dfs["credit_test_df"]
sales_test_df    = test_dfs["sales_test_df"]
billing_test_df  = test_dfs["billing_test_df"]
balance_test_df  = test_dfs["balance_test_df"]
channel_test_df  = test_dfs["channel_test_df"]
marketing_test_df= test_dfs["marketing_test_df"]
performance_test_df = test_dfs["performance_test_df"]

del test_dfs
gc.collect()

#### Train ####

train_df = customer_train_df.merge(credit_train_df, on=['기준년월', 'ID'], how='left')
print("Step1 저장 완료: train_step1, shape:", train_df.shape)
del customer_train_df, credit_train_df
gc.collect()

# 이후 merge할 데이터프레임 이름과 단계 정보를 리스트에 저장
merge_list = [
    ("sales_train_df",    "Step2"),
    ("billing_train_df",  "Step3"),
    ("balance_train_df",  "Step4"),
    ("channel_train_df",  "Step5"),
    ("marketing_train_df","Step6"),
    ("performance_train_df", "최종")
]

# 나머지 단계 merge
for df_name, step in merge_list:
    # globals()로 동적 변수 접근하여 merge 수행
    train_df = train_df.merge(globals()[df_name], on=['기준년월', 'ID'], how='left')
    print(f"{step} 저장 완료: train_{step}, shape:", train_df.shape)
    # 사용한 변수는 메모리 해제를 위해 삭제
    del globals()[df_name]
    gc.collect()

#### Test ####

test_df = customer_test_df.merge(credit_test_df, on=['기준년월', 'ID'], how='left')
print("Step1 저장 완료: test_step1, shape:", test_df.shape)
del customer_test_df, credit_test_df
gc.collect()

# 이후 merge할 데이터프레임 이름과 단계 정보를 리스트에 저장
merge_list = [
    ("sales_test_df",    "Step2"),
    ("billing_test_df",  "Step3"),
    ("balance_test_df",  "Step4"),
    ("channel_test_df",  "Step5"),
    ("marketing_test_df","Step6"),
    ("performance_test_df", "최종")
]

# 나머지 단계 merge
for df_name, step in merge_list:
    # globals()로 동적 변수 접근하여 merge 수행
    test_df = test_df.merge(globals()[df_name], on=['기준년월', 'ID'], how='left')
    print(f"{step} 저장 완료: test_{step}, shape:", test_df.shape)
    # 사용한 변수는 메모리 해제를 위해 삭제
    del globals()[df_name]
    gc.collect()

feature_cols = [col for col in train_df.columns if col not in ["ID", "Segment"]]

X = train_df[feature_cols].copy()
y = train_df["Segment"].copy()
test_ids = test_df["ID"].copy()
X_test = test_df[feature_cols].copy()
del train_df, test_df
gc.collect()

# 타깃 라벨 인코딩
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
del y
gc.collect()

categorical_features = X.select_dtypes(include=['object']).columns.tolist()

encoders = {}  # 각 컬럼별 encoder 저장

for col in categorical_features:
    le_train = LabelEncoder()
    X[col] = le_train.fit_transform(X[col])
    encoders[col] = le_train
    unseen_labels_val = set(X_test[col]) - set(le_train.classes_)
    if unseen_labels_val:
        le_train.classes_ = np.append(le_train.classes_, list(unseen_labels_val))
    X_test[col] = le_train.transform(X_test[col])
    del le_train
    gc.collect()

# 데이터 전처리
print("\n=== 데이터 전처리 시작 ===")
print(f"X shape: {X.shape}, X_test shape: {X_test.shape}")
print(f"X memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"X_test memory usage: {X_test.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 스케일링 전에 데이터 타입 변환
X = X.astype(np.float32)
X_test = X_test.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
print(f"X_scaled shape: {X_scaled.shape}, X_test_scaled shape: {X_test_scaled.shape}")
print(f"X_scaled memory usage: {X_scaled.nbytes / 1024**2:.2f} MB")
print(f"X_test_scaled memory usage: {X_test_scaled.nbytes / 1024**2:.2f} MB")
del X, X_test
gc.collect()

# train_test_split 적용
print("\n=== train_test_split 적용 ===")
print(f"y_encoded unique values: {np.unique(y_encoded)}")
print(f"y_encoded memory usage: {y_encoded.nbytes / 1024**2:.2f} MB")

# 메모리 사용량을 줄이기 위해 float32로 변환
X_scaled = X_scaled.astype(np.float32)

# train_test_split 수행
train_data, val_data, train_labels, val_labels = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"train_data shape: {train_data.shape}, val_data shape: {val_data.shape}")
print(f"train_labels shape: {train_labels.shape}, val_labels shape: {val_labels.shape}")
print(f"train_data memory usage: {train_data.nbytes / 1024**2:.2f} MB")
print(f"val_data memory usage: {val_data.nbytes / 1024**2:.2f} MB")
print(f"train_labels memory usage: {train_labels.nbytes / 1024**2:.2f} MB")
print(f"val_labels memory usage: {val_labels.nbytes / 1024**2:.2f} MB")

# 불필요한 데이터 삭제
del X_scaled, y_encoded
gc.collect()

# PyTorch 텐서로 변환
print("\n=== PyTorch 텐서 변환 ===")
train_tensor = torch.FloatTensor(train_data)
val_tensor = torch.FloatTensor(val_data)
test_tensor = torch.FloatTensor(X_test_scaled)
train_label_tensor = torch.LongTensor(train_labels)
val_label_tensor = torch.LongTensor(val_labels)

print(f"train_tensor shape: {train_tensor.shape}, val_tensor shape: {val_tensor.shape}")
print(f"test_tensor shape: {test_tensor.shape}")
print(f"train_label_tensor shape: {train_label_tensor.shape}, val_label_tensor shape: {val_label_tensor.shape}")
print(f"train_tensor memory usage: {train_tensor.element_size() * train_tensor.nelement() / 1024**2:.2f} MB")
print(f"val_tensor memory usage: {val_tensor.element_size() * val_tensor.nelement() / 1024**2:.2f} MB")
print(f"test_tensor memory usage: {test_tensor.element_size() * test_tensor.nelement() / 1024**2:.2f} MB")

# 원본 데이터 삭제
del train_data, val_data, train_labels, val_labels, X_test_scaled
gc.collect()

# 데이터셋 생성
print("\n=== 데이터셋 생성 ===")
train_dataset = CardDataset(train_tensor, train_label_tensor)
val_dataset = CardDataset(val_tensor, val_label_tensor)
print(f"train_dataset size: {len(train_dataset)}, val_dataset size: {len(val_dataset)}")
print(f"train_dataset labels unique values: {np.unique(train_dataset.labels)}")
del train_tensor, train_label_tensor, val_tensor, val_label_tensor
gc.collect()

# DataLoader 생성
print("\n=== DataLoader 생성 ===")
train_dataloader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=None
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=None
)
print(f"train_dataloader batches: {len(train_dataloader)}, val_dataloader batches: {len(val_dataloader)}")

# 딥러닝 모델 학습
print("\n=== 딥러닝 모델 학습 시작 ===")


# 재현성 보장을 위한 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 시드 설정
set_seed(42)

# 모델 저장 경로 설정
save_dir = 'checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# MPS 디바이스 확인
print(f"MPS 사용 가능: {torch.backends.mps.is_available()}")
print(f"MPS 사용 중: {torch.backends.mps.is_built()}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# ResidualBlock 클래스 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ResidualBlock, self).__init__()
        # Main branch
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        
        # Shortcut connection
        self.shortcut = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        # Main branch
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        # Residual connection
        out = out + identity
        return out

# SimpleModel 클래스 정의
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=5):
        super(SimpleModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        
        # Residual blocks with gradual dimension reduction
        self.res1 = ResidualBlock(hidden_dim, hidden_dim // 2, hidden_dim // 2)
        self.res2 = ResidualBlock(hidden_dim // 2, hidden_dim // 4, hidden_dim // 4)
        self.res3 = ResidualBlock(hidden_dim // 4, hidden_dim // 8, hidden_dim // 8)
        
        # Output layers with gradual dimension reduction
        self.fc1 = nn.Linear(hidden_dim // 8, hidden_dim // 16)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 16)
        self.fc2 = nn.Linear(hidden_dim // 16, hidden_dim // 32)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 32)
        self.fc3 = nn.Linear(hidden_dim // 32, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Input processing
        x = self.input_fc(x)
        x = self.bn_input(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Output layers
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

# 데이터 증강 함수들
def add_noise(x, noise_factor=0.1):
    noise = torch.randn_like(x) * noise_factor
    return x + noise

def add_gaussian_noise(x, mean=0.0, std=0.05):
    noise = torch.randn_like(x) * std + mean
    return x + noise

def add_dropout_noise(x, p=0.05):
    mask = torch.bernoulli(torch.ones_like(x) * (1-p))
    return x * mask

def augment_data(x):
    x = add_noise(x)
    x = add_gaussian_noise(x)
    x = add_dropout_noise(x)
    return x

# 평가 함수
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    f1 = f1_score(all_targets, all_preds, average='weighted')
    avg_loss = total_loss / len(dataloader)
    return avg_loss, f1

# 손실 함수 클래스들
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=4.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.2):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_classes - 1)
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        loss = (-smooth_one_hot * log_probs).sum(dim=-1).mean()
        return loss

# 딥러닝 모델 학습
def train_deep_model():
    print("\n=== 모델 초기화 ===")
    # TensorBoard 설정
    writer = SummaryWriter('runs/deep_learning_experiment')
    
    # 모델 설정
    input_dim = train_dataset.features.shape[1]
    output_dim = len(np.unique(train_dataset.labels))
    print(f"input_dim: {input_dim}, output_dim: {output_dim}")
    model = SimpleModel(input_dim=input_dim, output_dim=output_dim).to(device)
    print(f"Model structure:\n{model}")
    
    # 손실 함수 설정
    criterion_ce = LabelSmoothingLoss(smoothing=0.2)
    criterion_focal = FocalLoss(alpha=0.25, gamma=4.0)
    loss_weights = {'ce': 0.3, 'focal': 0.7}
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.02)
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")
    
    # Learning rate warmup 스케줄러
    num_warmup_steps = 15
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=num_warmup_steps,
        verbose=True
    )
    
    # ReduceLROnPlateau 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # Early Stopping 설정
    early_stopping_patience = 20
    early_stopping_counter = 0
    best_val_f1 = 0.0

    # Gradient accumulation 설정
    accumulation_steps = 8
    optimizer.zero_grad()

    # 학습 루프
    num_epochs = 200
    start_time = time.time()

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        # 학습
        model.train()
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_focal_loss = 0.0
        epoch_preds = []
        epoch_targets = []
        batch_count = 0
        
        try:
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                try:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # 데이터 증강
                    inputs = augment_data(inputs)
                    
                    outputs = model(inputs)
                    
                    # Combined loss 계산
                    loss_ce = criterion_ce(outputs, targets)
                    loss_focal = criterion_focal(outputs, targets)
                    loss = loss_weights['ce'] * loss_ce + loss_weights['focal'] * loss_focal
                    
                    # Gradient accumulation
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item() * accumulation_steps
                    epoch_ce_loss += loss_ce.item()
                    epoch_focal_loss += loss_focal.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_preds.extend(predicted.cpu().numpy())
                    epoch_targets.extend(targets.cpu().numpy())
                    batch_count += 1
                    
                    # 배치 처리 후 메모리 정리
                    del inputs, targets, outputs, loss, loss_ce, loss_focal, predicted
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                                    
                except Exception as e:
                    print(f"배치 처리 중 오류 발생: {e}")
                    continue
        
        except Exception as e:
            print(f"에폭 처리 중 오류 발생: {e}")
            continue
        
        # 검증
        val_loss, val_f1 = evaluate(model, val_dataloader, criterion_ce, device)
        
        if batch_count > 0:
            train_f1 = f1_score(epoch_targets, epoch_preds, average='weighted')
            
            # TensorBoard에 메트릭 기록
            writer.add_scalar('Loss/train', epoch_loss/batch_count, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Loss/ce', epoch_ce_loss/batch_count, epoch)
            writer.add_scalar('Loss/focal', epoch_focal_loss/batch_count, epoch)
            writer.add_scalar('F1/train', train_f1, epoch)
            writer.add_scalar('F1/validation', val_f1, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            print(f"\nEpoch {epoch+1} 완료:")
            print(f"Train Loss: {epoch_loss/batch_count:.4f}")
            print(f"Train F1: {train_f1:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation F1: {val_f1:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Learning rate 스케줄러 업데이트
            if epoch < num_warmup_steps:
                warmup_scheduler.step()
            else:
                scheduler.step(val_f1)
            
            # 최고 검증 F1 저장
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                early_stopping_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_f1': best_val_f1,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"새로운 최고 검증 F1: {best_val_f1:.4f}")
            else:
                early_stopping_counter += 1
                print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered!")
                    break
        
        # 에폭 처리 후 메모리 정리
        del epoch_loss, epoch_ce_loss, epoch_focal_loss, epoch_preds, epoch_targets
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        elapsed_time = time.time() - start_time
        print(f"경과 시간: {elapsed_time/60:.2f}분")

    # TensorBoard writer 닫기
    writer.close()

    print("\n=== 학습 완료 ===")
    total_time = time.time() - start_time
    print(f"총 학습 시간: {total_time/60:.2f}분")
    print(f"최고 검증 F1: {best_val_f1:.4f}")
    
    return model

# 딥러닝 모델 학습 및 예측
print("\n=== 딥러닝 모델 학습 시작 ===")
deep_model = train_deep_model()

# 최고 성능 모델 로드
checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
deep_model.load_state_dict(checkpoint['model_state_dict'])
del checkpoint
gc.collect()

# 예측 수행
deep_model.eval()
with torch.no_grad():
    test_tensor = test_tensor.to(device)
    prediction = deep_model(test_tensor)
    del test_tensor
    gc.collect()

# 예측 결과 변환
prediction = prediction.cpu().numpy()
prediction = prediction.argmax(axis=1)
prediction = le_target.inverse_transform(prediction)

# 딥러닝 결과 저장
test_data = pd.DataFrame({
    'ID': test_ids.values,
    'pred_label': prediction
})
del prediction, test_ids
gc.collect()

submission_deep = test_data.groupby("ID")["pred_label"] \
    .agg(lambda x: x.value_counts().idxmax()) \
    .reset_index()

submission_deep.columns = ["ID", "Segment"]
submission_deep.to_csv('./deep_submit.csv', index=False)
del test_data, submission_deep
gc.collect()

print("\n=== 딥러닝 모델 학습 완료 ===")