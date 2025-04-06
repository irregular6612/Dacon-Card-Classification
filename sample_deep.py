import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
import multiprocessing
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torchvision.ops import sigmoid_focal_loss
from torch.utils.tensorboard import SummaryWriter
import os

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

# CPU 코어 수 확인 및 워커 수 제한
num_workers = min(4, multiprocessing.cpu_count())
print(f"사용 가능한 CPU 코어 수: {multiprocessing.cpu_count()}")
print(f"실제 사용할 워커 수: {num_workers}")

# 커스텀 데이터셋 정의
class CardDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def dataLoadAndPreprocess():
    # 데이터 로드 (CPU에서 수행)
    train_df = pd.read_csv('train_pca_df.csv').values
    test_df = pd.read_csv('test_pca_df.csv').values
    label_df = pd.read_csv('target_train_df.csv').values.reshape(-1, 1)
    
    # 레이블 인코딩
    le = LabelEncoder()
    label_df = le.fit_transform(label_df)
    
    # 스케일링 적용
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    # train_test_split 적용
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_scaled, label_df, test_size=0.2, random_state=42, stratify=label_df
    )
    
    # PyTorch 텐서로 변환 (CPU에서 수행)
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)
    test_tensor = torch.FloatTensor(test_scaled)
    train_label_tensor = torch.LongTensor(train_labels)
    val_label_tensor = torch.LongTensor(val_labels)
    
    return train_tensor, val_tensor, test_tensor, train_label_tensor, val_label_tensor, le

print("데이터 로드 중...")
train_tensor, val_tensor, test_tensor, train_label_tensor, val_label_tensor, le = dataLoadAndPreprocess()
print("데이터 로드 완료")

print(train_tensor.shape)
print(val_tensor.shape)
print(test_tensor.shape)
print(train_label_tensor.shape)
print(val_label_tensor.shape)

# 데이터셋 생성
train_dataset = CardDataset(train_tensor, train_label_tensor)
val_dataset = CardDataset(val_tensor, val_label_tensor)

# DataLoader 생성
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

# 간단한 모델 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        identity = x
        out = self.dropout(self.relu(self.bn1(self.fc1(x))))
        out = self.dropout(self.relu(self.bn2(self.fc2(out))))
        return out + identity

class SimpleModel(nn.Module):
    def __init__(self, input_dim=812, hidden_dim=512, output_dim=5):
        super(SimpleModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.res3 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.bn_input(self.input_fc(x))))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

def add_noise(x, noise_factor=0.05):
    noise = torch.randn_like(x) * noise_factor
    return x + noise

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def calculate_class_weights(labels):
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

def main():
    # 전역 변수 선언
    global test_tensor, le
    
    # TensorBoard 설정
    writer = SummaryWriter('runs/deep_learning_experiment')
    
    # 모델, 손실 함수, 옵티마이저 설정
    print("\n=== 모델 설정 ===")
    model = SimpleModel(input_dim=train_tensor.shape[1], output_dim=len(torch.unique(train_label_tensor))).to(device)
    
    # 클래스 가중치 계산
    class_weights = calculate_class_weights(train_label_tensor.numpy())
    class_weights = class_weights.to(device)
    
    # 손실 함수 설정
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_focal = FocalLoss(alpha=1, gamma=2.5)
    
    # 손실 함수 가중치 설정
    loss_weights = {'ce': 0.2, 'focal': 0.8}
    
    # L2 정규화를 포함한 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Early Stopping 설정
    early_stopping_patience = 5
    early_stopping_counter = 0
    best_val_f1 = 0.0

    # 학습 루프
    print("\n=== 학습 시작 ===")
    num_epochs = 100
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
                    
                    # 데이터 증강: 노이즈 추가
                    inputs = add_noise(inputs)
                    
                    outputs = model(inputs)
                    
                    # Combined loss 계산
                    loss_ce = criterion_ce(outputs, targets)
                    loss_focal = criterion_focal(outputs, targets)
                    loss = loss_weights['ce'] * loss_ce + loss_weights['focal'] * loss_focal
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping 추가
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_ce_loss += loss_ce.item()
                    epoch_focal_loss += loss_focal.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_preds.extend(predicted.cpu().numpy())
                    epoch_targets.extend(targets.cpu().numpy())
                    batch_count += 1
                                    
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
            
            # 모델 파라미터 분포 시각화
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'Parameters/{name}', param.data, epoch)
            
            print(f"\nEpoch {epoch+1} 완료:")
            print(f"Train Loss: {epoch_loss/batch_count:.4f}")
            print(f"Train F1: {train_f1:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation F1: {val_f1:.4f}")
            
            # 학습률 스케줄러 업데이트
            scheduler.step(val_f1)
            
            # 최고 검증 F1 저장
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                early_stopping_counter = 0
                # 모델 체크포인트 저장
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
                
                # Early stopping 체크
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered!")
                    break
        
        elapsed_time = time.time() - start_time
        print(f"경과 시간: {elapsed_time/60:.2f}분")

    # TensorBoard writer 닫기
    writer.close()

    print("\n=== 학습 완료 ===")
    total_time = time.time() - start_time
    print(f"총 학습 시간: {total_time/60:.2f}분")
    print(f"최고 검증 F1: {best_val_f1:.4f}")
    
    # 최고 성능 모델 로드
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 예측 수행
    model.eval()
    with torch.no_grad():
        # 테스트 데이터를 MPS 디바이스로 이동
        test_tensor = test_tensor.to(device)
        prediction = model(test_tensor)

    # MPS 텐서를 CPU로 이동한 후 NumPy 배열로 변환
    prediction = prediction.cpu().numpy()
    prediction = prediction.argmax(axis=1)
    prediction = le.inverse_transform(prediction)

    submission = pd.read_csv('./datasets/sample_submission.csv')
    submission['Segment'] = prediction
    submission.to_csv('deep-submission.csv', index=False)

if __name__ == '__main__':
    main()
