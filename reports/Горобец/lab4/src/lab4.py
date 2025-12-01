import pandas as pd              # Импорт библиотеки pandas для работы с таблицами (DataFrame)
import torch                     # Импорт PyTorch для построения и обучения нейросети
import torch.nn as nn            # Модуль для описания архитектуры нейросети
import torch.optim as optim      # Модуль для оптимизаторов (Adam, SGD и др.)
from sklearn.model_selection import train_test_split   # Функция для разделения данных на train/test
from sklearn.preprocessing import StandardScaler       # Стандартизация признаков (нормализация)
from sklearn.metrics import accuracy_score, f1_score   # Метрики качества классификации

# 1. Загрузка и подготовка данных
df = pd.read_csv("C:/Users/Anton/Downloads/winequality-white.csv", sep=";")  # Загружаем датасет вина
df["Target"] = (df["quality"] >= 7).astype(int)   # Создаём бинарную цель: 1 если качество >=7, иначе 0
X = df.drop(["quality", "Target"], axis=1)        # Признаки: все столбцы кроме quality и Target
y = df["Target"]                                  # Целевая переменная (классификация)

scaler = StandardScaler()                         # Инициализация стандартизатора
X_scaled = scaler.fit_transform(X)                # Масштабируем признаки (среднее=0, дисперсия=1)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# Делим данные: 70% обучение, 30% тест

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)   # Признаки train → тензор float32
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)     # Признаки test → тензор float32
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Цель train → столбец
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)    # Цель test → столбец

# 2. Архитектура нейронной сети
class MLP(nn.Module):                               # Определяем класс многослойного перцептрона
    def __init__(self):
        super(MLP, self).__init__()                  # Наследуем от nn.Module
        self.model = nn.Sequential(                  # Последовательная модель
            nn.Linear(X_train.shape[1], 12),         # Входной слой: число признаков → 12 нейронов
            nn.ReLU(),                               # Активация ReLU
            nn.Linear(12, 12),                       # Скрытый слой: 12 → 12 нейронов
            nn.ReLU(),                               # Активация ReLU
            nn.Linear(12, 1)                         # Выходной слой: 1 нейрон (вероятность класса)
        )

    def forward(self, x):                            # Метод прямого прохода
        return self.model(x)                         # Прогоняем данные через слои

# 3. Инициализация модели, функции потерь и оптимизатора
model = MLP()                                       # Создаём экземпляр модели
criterion = nn.BCEWithLogitsLoss()                  # Функция потерь для бинарной классификации
optimizer = optim.Adam(model.parameters(), lr=0.001) # Оптимизатор Adam с шагом обучения 0.001

# 4. Цикл обучения
epochs = 100  # увеличено вдвое для эксперимента   # Количество эпох обучения

for epoch in range(epochs):                         # Цикл по эпохам
    model.train()                                   # Переводим модель в режим обучения
    y_pred = model(X_train_tensor)                  # Получаем предсказания на train
    loss = criterion(y_pred, y_train_tensor)        # Считаем функцию потерь

    optimizer.zero_grad()                           # Обнуляем градиенты
    loss.backward()                                 # Считаем градиенты (обратное распространение)
    optimizer.step()                                # Обновляем веса модели

    if (epoch + 1) % 10 == 0:                       # Каждые 10 эпох печатаем loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 5. Оценка модели
model.eval()                                        # Переводим модель в режим оценки
with torch.no_grad():                               # Отключаем вычисление градиентов
    y_logits = model(X_test_tensor)                 # Получаем логиты на тесте
    y_pred = torch.sigmoid(y_logits)                # Применяем сигмоиду → вероятности
    y_pred_class = (y_pred > 0.5).int()             # Превращаем вероятности в классы (0/1)

y_true = y_test_tensor.numpy()                      # Истинные значения → numpy
y_pred_np = y_pred_class.numpy()                    # Предсказания → numpy

acc = accuracy_score(y_true, y_pred_np)             # Считаем Accuracy
f1 = f1_score(y_true, y_pred_np)                    # Считаем F1-score

print(f"Accuracy: {acc:.4f}")                       # Выводим точность
print(f"F1-score: {f1:.4f}")                        # Выводим F1
