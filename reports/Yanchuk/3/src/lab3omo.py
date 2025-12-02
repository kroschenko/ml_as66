import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("encoder", ColumnTransformer(
            transformers=[("cat", encoder, categorical_features)],
            remainder="drop"
        )),
        ("clf", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = (pipe, acc)

print("Точность моделей:")
for name, (_, acc) in results.items():
    print(f"{name}: {acc:.4f}")

dt_model = results["DecisionTree"][0].named_steps["clf"]
ohe = results["DecisionTree"][0].named_steps["encoder"].named_transformers_["cat"]

feature_names = ohe.get_feature_names_out(categorical_features)
importances = dt_model.feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nТоп-10 признаков по важности (DecisionTree):")
print(feat_imp.head(10))
