import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score, accuracy_score

path = "kddcup.data_10_percent_corrected"
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
]
df = pd.read_csv(path, names=columns)

# normal 0, остальные 1
df["target"] = (df["label"] != "normal.").astype(int)

# признаки
cat_cols = ["protocol_type", "service", "flag"]
df = pd.get_dummies(df.drop(columns=["label"]), columns=cat_cols)

# подвыборка 5к
sample_size = 5000
sample_size = min(sample_size, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

X = df_sample.drop(columns=["target"])
y = df_sample["target"]

# Разбиение на обучающию и тестовую
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

ks = [1, 3, 5, 7]
results = []

# k-NN
for k in ks:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, n_jobs=1))
    ])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    y_pred = pipe.predict(X_test)
    pred_time = time.time() - t0

    recall = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    results.append({
        "model": "k-NN",
        "param": f"k={k}",
        "recall_attack": round(recall, 4),
        "accuracy": round(acc, 4),
        "fit_time_s": round(fit_time, 4),
        "predict_time_s": round(pred_time, 4)
    })

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
t0 = time.time()
dt.fit(X_train, y_train)
fit_time = time.time() - t0

t0 = time.time()
y_pred = dt.predict(X_test)
pred_time = time.time() - t0

recall = recall_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

results.append({
    "model": "Decision Tree",
    "param": "",
    "recall_attack": round(recall, 4),
    "accuracy": round(acc, 4),
    "fit_time_s": round(fit_time, 4),
    "predict_time_s": round(pred_time, 4)
})

# Linear SVM
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(max_iter=5000, random_state=42))
])

t0 = time.time()
svm_pipe.fit(X_train, y_train)
fit_time = time.time() - t0

t0 = time.time()
y_pred = svm_pipe.predict(X_test)
pred_time = time.time() - t0

recall = recall_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

results.append({
    "model": "Linear SVM",
    "param": "",
    "recall_attack": round(recall, 4),
    "accuracy": round(acc, 4),
    "fit_time_s": round(fit_time, 4),
    "predict_time_s": round(pred_time, 4)
})


results_df = pd.DataFrame(results)
print("Результаты:")
print(results_df.to_string(index=False))
best_by_recall = results_df.loc[results_df["recall_attack"].idxmax()]
best_by_fit = results_df.loc[results_df["fit_time_s"].idxmin()]
best_by_pred = results_df.loc[results_df["predict_time_s"].idxmin()]

print("\nКороткий анализ:")
print(f"- Лучшая по recall: {best_by_recall['model']} {best_by_recall['param']} (recall={best_by_recall['recall_attack']})")
print(f"- Быстрее всего обучается: {best_by_fit['model']} {best_by_fit['param']} (fit_time={best_by_fit['fit_time_s']}s)")
print(f"- Быстрее всего предсказывает: {best_by_pred['model']} {best_by_pred['param']} (predict_time={best_by_pred['predict_time_s']}s)")
