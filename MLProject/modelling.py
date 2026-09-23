# modelling.py (versi MLProject untuk workflow CI)
# sama seperti modelling.py di Membangun_model, ditambah penulisan run_id
# ke file supaya bisa dipakai step docker di workflow.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

mlflow.sklearn.autolog()

# set_experiment sengaja tidak dipakai: mlflow run sudah menyiapkan run-nya sendiri

# dataset hasil preprocessing, satu folder dengan file ini
data = pd.read_csv("telco_churn_preprocessing.csv")

X = data.drop("Churn", axis=1).astype(float)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    akurasi = model.score(X_test, y_test)
    print("akurasi di data uji:", round(akurasi, 4))

    # run id disimpan supaya step workflow berikutnya
    # (build image docker) tahu model hasil run yang mana
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

print("training CI selesai")
