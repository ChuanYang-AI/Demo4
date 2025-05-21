# Chicago Taxi Vertex AI 全自动流水线

本项目实现了芝加哥出租车数据的端到端自动化机器学习流水线，支持**本地一键运行**，适合开发、调试、生产级 MLOps 场景。

---

## 目录结构

```
.
├── taxi_dataflow_clean.py           # Dataflow 脚本，负责数据清洗
├── taxi_vertex_pipeline.py          # Vertex AI/KFP Pipeline 主体
├── README.md                        # 本说明文档
├── README-GCP.pdf                   # 基于 GCP 的详细说明文档
```

---

## 各文件用途与使用场景

- **taxi_dataflow_clean.py**
  - 作用：Apache Beam Dataflow 脚本，负责从 BigQuery 读取原始数据，清洗并写入新 BigQuery 表。
  - 使用场景：作为 pipeline 的输入文件，由 pipeline 自动上传到 GCS 并用于 Dataflow 作业。
  - 支持参数通过命令行或环境变量传递。

- **taxi_vertex_pipeline.py**
  - 作用：定义了完整的 Vertex AI/Kubeflow Pipeline，包括：
    - 自动上传 taxi_dataflow_clean.py
    - Dataflow 数据清洗
    - 训练（sklearn 线性回归）
    - 模型上传到 GCS
    - 自动创建 Endpoint 并部署模型
    - 自动推理
  - 使用场景：
    - 本地一键全自动运行（推荐开发/调试/实验）
  - 所有参数均支持通过环境变量配置。

---

## 环境变量优先级说明

- 所有参数均可通过环境变量设置，优先级为：
  1. 命令行参数（本地运行）
  2. 环境变量（变量名为大写，如 PROJECT_ID）
  3. 默认值（如有）

---

## 本地一键全自动运行（适合开发/调试/实验）

1. **准备好本地 taxi_dataflow_clean.py 脚本**（内容已给出，无需修改）。
2. **设置环境变量并运行 taxi_vertex_pipeline.py**：

```bash
export GOOGLE_CLOUD_PROJECT=cy-aispeci-demo
export PROJECT_ID=cy-aispeci-demo
export REGION=us-central1
export BUCKET_NAME=cy-aispeci-demo-bucket-zp   # 你的 GCS bucket 名称
export TEMP_LOCATION=gs://$BUCKET_NAME/temp
export DATAFLOW_CLEAN_LOCAL_PATH=./taxi_dataflow_clean.py
export GCS_SCRIPT_DIR=gs://$BUCKET_NAME/scripts/
export BQ_OUTPUT_TABLE=cy-aispeci-demo.chicago_taxi_dataset.chicago_taxi_cleaned
export GCS_MODEL_DIR=gs://$BUCKET_NAME/model_dir
export GCS_MODEL_PATH=gs://$BUCKET_NAME/model_dir/model.pkl
export PIPELINE_ROOT=gs://$BUCKET_NAME/pipeline_root
export MACHINE_TYPE=n1-standard-4
export MIN_REPLICA_COUNT=1
export MAX_REPLICA_COUNT=3
export ENDPOINT_DISPLAY_NAME=chicago-taxi-endpoint

python taxi_vertex_pipeline.py
```

- 无需手动修改 py 文件参数，直接用环境变量即可。
- 本地会自动上传本地的 Dataflow 脚本和参数，然后在云端执行所有后续步骤。
- 全流程自动完成：脚本上传、Dataflow 清洗、训练、上传、部署、推理。

---

## 主要参数说明

- `project` / `PROJECT_ID`：GCP 项目ID（如 cy-aispeci-demo）
- `region` / `REGION`：GCP 区域（如 us-central1）
- `bucket_name` / `BUCKET_NAME`：你的 GCS bucket 名称（如 cy-aispeci-demo-bucket-zp）
- `temp_location` / `TEMP_LOCATION`：GCS 临时目录（如 gs://$BUCKET_NAME/temp）
- `dataflow_clean_local_path` / `DATAFLOW_CLEAN_LOCAL_PATH`：
  - 本地一键运行时：填写本地实际路径（如 ./taxi_dataflow_clean.py 或绝对路径）
  - 云函数 HTTP 触发时：填写 /workspace/taxi_dataflow_clean.py（云函数部署时自动放在该目录）
- `gcs_script_dir` / `GCS_SCRIPT_DIR`：GCS 目录，自动上传 taxi_dataflow_clean.py（如 gs://$BUCKET_NAME/scripts/）
- `bq_output_table` / `BQ_OUTPUT_TABLE`：清洗后 BigQuery 表名（如 cy-aispeci-demo.chicago_taxi_dataset.chicago_taxi_cleaned）
- `gcs_model_dir` / `GCS_MODEL_DIR`：GCS 目录，模型存储（如 gs://$BUCKET_NAME/model_dir）
- `pipeline_root` / `PIPELINE_ROOT`：Vertex AI Pipeline 根目录（可选，如 gs://$BUCKET_NAME/pipeline_root）

---

### 如何获取/创建 GCS bucket？

1. 控制台网页创建：[Google Cloud Console](https://console.cloud.google.com/storage/browser)
2. 或命令行创建：

```bash
export PROJECT_ID=cy-aispeci-demo
export BUCKET_NAME=cy-aispeci-demo-bucket-zp
export REGION=us-central1
gcloud auth application-default set-quota-project $PROJECT_ID
gcloud config set project $PROJECT_ID
# 创建 bucket（区域建议与 Vertex AI 区域一致）
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME
```
4. 查看已有 bucket：
```bash
gsutil ls
```
