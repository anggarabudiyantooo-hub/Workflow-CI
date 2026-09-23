# Workflow-CI

Workflow CI untuk re-training model klasifikasi churn (Telco Customer Churn) memakai MLflow Project. Submission kelas Membangun Sistem Machine Learning (Dicoding).

## Isi repo

- `MLProject/` - project MLflow: `modelling.py`, `conda.yaml`, `MLProject`, dataset hasil preprocessing
- `.github/workflows/main.yml`:
  1. `mlflow run MLProject` tiap ada push,
  2. artefak hasil training disimpan sebagai artifact repo ini,
  3. `mlflow models build-docker` lalu push image ke Docker Hub.

## Secrets

Settings → Secrets and variables → Actions:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Step docker di-skip otomatis bila secrets belum diisi.
