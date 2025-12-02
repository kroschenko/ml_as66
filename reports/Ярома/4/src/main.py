import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


DATA_PATH = "kddcup.data_10_percent_corrected"
SAMPLE_SIZE = 5000        # размер подвыборки
TEST_SIZE = 0.3            # доля теста
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_STATE)


columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
]

print("Загрузка данных...")
df = pd.read_csv(DATA_PATH, names=columns)
print("Исходный размер:", df.shape)

# бинарная цель
df["target"] = (df["label"] != "normal.").astype(int)

# кодирование категориальных признаков
cat_cols = ["protocol_type", "service", "flag"]
df_num = pd.get_dummies(df.drop(columns=["label"]), columns=cat_cols)

# подвыборка
if SAMPLE_SIZE < len(df_num):
    df_sample, _ = train_test_split(df_num, train_size=SAMPLE_SIZE, stratify=df_num["target"], random_state=RANDOM_STATE)
else:
    df_sample = df_num.copy()

print("Размер подвыборки:", df_sample.shape)

X = df_sample.drop(columns=["target"]).values
y = df_sample["target"].values

# стандартизация
scaler = StandardScaler()
X = scaler.fit_transform(X)

# разбиение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train class balance:", np.bincount(y_train))
print("Test class balance:", np.bincount(y_test))

# преобразование в тензоры PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape (N,1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# Определение модели
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, p_dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, 1)  # логит для бинарной задачи
        )

    def forward(self, x):
        return self.net(x)  # возвращаем логиты (без сигмоида)


# Функции обучения и оценки
def train_model(hidden_size, epochs=EPOCHS, lr=LR, verbose=False):
    input_dim = X_train.shape[1]
    model = MLP(input_dim=input_dim, hidden_dim=hidden_size).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"loss": []}
    start_time = time.time()

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_loader.dataset)
        history["loss"].append(epoch_loss)
        if verbose:
            print(f"Epoch {epoch}/{epochs} — loss: {epoch_loss:.6f}")

    train_time = time.time() - start_time

    # Оценка на тесте
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds_batch = (probs > 0.5).float().cpu().numpy().reshape(-1)
            true_batch = yb.cpu().numpy().reshape(-1)
            preds.append(preds_batch)
            targets.append(true_batch)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    recall = recall_score(targets, preds)
    precision = precision_score(targets, preds)

    return {
        "hidden_size": hidden_size,
        "train_time_s": train_time,
        "accuracy": acc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "loss_history": history["loss"]
    }

# Запуск  hidden=64 и hidden=16
hidden_list = [64, 16]
results = []

for h in hidden_list:
    print(f"\n--- Обучение MLP с hidden_size = {h} ---")
    res = train_model(hidden_size=h, epochs=EPOCHS, lr=LR, verbose=True)
    results.append(res)
    print(f"hidden={h} — train_time: {res['train_time_s']:.2f}s, accuracy: {res['accuracy']:.4f}, f1: {res['f1']:.4f}, recall: {res['recall']:.4f}, precision: {res['precision']:.4f}")


print("\n=== Сравнение результатов ===")
for r in results:
    print(f"hidden={r['hidden_size']:2d} | time={r['train_time_s']:.2f}s | acc={r['accuracy']:.4f} | f1={r['f1']:.4f} | recall={r['recall']:.4f} | prec={r['precision']:.4f}")

h64 = next(r for r in results if r["hidden_size"] == 64)
h16 = next(r for r in results if r["hidden_size"] == 16)

print("\nКраткие выводы:")
print(f"- При переходе 64 -> 16 нейронов: время обучения {'уменьшилось' if h16['train_time_s'] < h64['train_time_s'] else 'увеличилось'} "
      f"({h64['train_time_s']:.2f}s -> {h16['train_time_s']:.2f}s).")
print(f"- Точность (accuracy): {h64['accuracy']:.4f} -> {h16['accuracy']:.4f}")
print(f"- F1-score: {h64['f1']:.4f} -> {h16['f1']:.4f}")
print(f"- Recall (важно для детекции атак): {h64['recall']:.4f} -> {h16['recall']:.4f}")

print("\nЗамечание: результаты зависят от размера подвыборки, числа эпох, батча и random seed.")
