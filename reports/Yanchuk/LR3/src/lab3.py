import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("car_evaluation.csv", header=None)
df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_features = X.columns.tolist()
encoder = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[("cat", encoder, categorical_features)],
    remainder="drop"
)

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42)
}

results = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("encoder", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

k_values = [1, 3, 5, 7, 9, 11, 15]
knn_scores = {}

for k in k_values:
    pipe_knn = Pipeline(steps=[
        ("encoder", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", KNeighborsClassifier(n_neighbors=k))
    ])
    pipe_knn.fit(X_train, y_train)
    y_pred = pipe_knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    knn_scores[k] = acc

best_k = max(knn_scores, key=knn_scores.get)
results[f"kNN (best k={best_k})"] = knn_scores[best_k]

print("Точность моделей:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

print("\nТочность k-NN при разных k:")
for k, acc in knn_scores.items():
    print(f"k={k}: {acc:.4f}")

dt_pipe = Pipeline(steps=[
    ("encoder", preprocessor),
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", DecisionTreeClassifier(random_state=42))
])
dt_pipe.fit(X_train, y_train)

dt_model = dt_pipe.named_steps["clf"]
ohe = dt_pipe.named_steps["encoder"].named_transformers_["cat"]

feature_names = ohe.get_feature_names_out(categorical_features)
importances = dt_model.feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nТоп-10 признаков по важности (DecisionTree):")
print(feat_imp.head(10))


