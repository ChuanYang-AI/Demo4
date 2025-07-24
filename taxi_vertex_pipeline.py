#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhipeng
# @Email: zhangzhipeng@polymericcloud.com
# @Date:   2025-05-14 14:38:24
# @Last Modified By:    zhipeng
# @Last Modified: 2025-05-20 17:42:01


import os
import time

from kfp import dsl
from kfp import compiler
# from kfp.dsl import importer, component, Output, Model, Artifact, OutputPath

from google.cloud import aiplatform
from google.cloud import storage
from google.cloud import bigquery

from google_cloud_pipeline_components.v1.dataflow import DataflowPythonJobOp
from google_cloud_pipeline_components.v1.wait_gcp_resources import WaitGcpResourcesOp
from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp, EndpointCreateOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.types import artifact_types


def get_env(key, default=None, required=False):
    v = os.environ.get(key, default)
    if required and v is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return v

def ensure_bq_dataset_exists(project, dataset, region):
    client = bigquery.Client(project=project, location=region)
    dataset_ref = client.dataset(dataset)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {project}.{dataset} already exists.")
    except Exception:
        client.create_dataset(bigquery.Dataset(dataset_ref))
        print(f"Created dataset {project}.{dataset}.")

@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "pandas", 
        "scikit-learn==1.5.2", 
        "google-cloud-bigquery", 
        "google-cloud-storage",
        "db-dtypes",
        "google-cloud-bigquery-storage"
    ],
)
def train_and_upload_model(
    bq_table: str,
    gcs_model_path: str,
    project: str,
    region: str,
    train_size: float = 0.8,
    test_size: float = 0.2,
) -> str:
    import pandas as pd
    from google.cloud import bigquery, storage
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import pickle

    import os
    def upload_file_to_gcs(local_path, gcs_path, project):
        bucket_name = gcs_path.replace('gs://', '').split('/')[0]
        prefix = '/'.join(gcs_path.replace('gs://', '').split('/')[1:])
        storage_client = storage.Client(project=project)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(prefix)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to {gcs_model_path}")

    client = bigquery.Client(project=project, location=region)
    df = client.query(f"SELECT * FROM `{bq_table}`").to_dataframe()
    print("训练数据 shape:", df.shape)
    print("训练数据前5行:\n", df.head())

    X = df[["trip_seconds", "trip_miles", "payment_type", "pickup_community_area", "dropoff_community_area"]]
    y = df["trip_total"]
    print("X columns:", X.columns)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("y head:", y.head())

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("模型系数:", getattr(model, 'coef_', None))
    print("模型截距:", getattr(model, 'intercept_', None))

    # 验证集评估
    from sklearn.metrics import r2_score, mean_squared_error
    y_val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    print(f"验证集 R2-score: {val_r2:.4f}, 验证集 RMSE: {val_rmse:.4f}")

    # 训练集评估
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    print(f"训练集 R2-score: {train_r2:.4f}, 训练集 RMSE: {train_rmse:.4f}")

    local_path = os.path.join("/tmp", "model.pkl")
    with open(local_path, "wb") as file:
        pickle.dump(model, file)
    print("本地模型文件大小:", os.path.getsize(local_path))
    print("准备上传的本地模型路径:", local_path)
    print("目标 GCS 路径:", gcs_model_path)
    upload_file_to_gcs(local_path, gcs_model_path, project)
    return gcs_model_path

@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "pandas", 
        "google-cloud-storage", 
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "scikit-learn==1.5.2", 
        "db-dtypes",
    ],
)
def download_and_predict(
    gcs_model_path: str,
    bq_table: str,
    project: str,
    region: str,
):
    import pandas as pd
    import pickle
    from google.cloud import storage, bigquery
    import os
    storage_client = storage.Client(project=project)
    bucket_name = gcs_model_path.replace('gs://', '').split('/')[0]
    blob_name = gcs_model_path.replace('gs://', '').split('/', 1)[-1]
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    model_local = "/tmp/model.pkl"
    blob.download_to_filename(model_local)
    with open(model_local, "rb") as file:
        model = pickle.load(file)
    bq_client = bigquery.Client(project=project, location=region)
    df = bq_client.query(f"SELECT * FROM `{bq_table}` LIMIT 10").to_dataframe()
    X = df[["trip_seconds", "trip_miles", "payment_type", "pickup_community_area", "dropoff_community_area"]]
    preds = model.predict(X)
    print('Sample predictions:', preds[:5])

@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery",
        "google-auth",
        "requests",
        "pandas",
        "db-dtypes",
        'google-cloud-pipeline-components'
    ],
)
def predict_via_endpoint(
    project: str,
    region: str,
    endpoint: dsl.Input[artifact_types.VertexEndpoint],
    bq_table: str,
    access_token: str = "",
):
    import pandas as pd
    from google.cloud import bigquery
    import requests
    import google.auth
    import google.auth.transport.requests

    # 获取 access token（如果没传入）
    if not access_token:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())
        access_token = credentials.token

    # 读取数据
    client = bigquery.Client(project=project, location=region)
    df = client.query(f"SELECT * FROM `{bq_table}` LIMIT 5").to_dataframe()
    print("Sample input:\n", df.head())

    # 构造请求体
    instances = df[["trip_seconds", "trip_miles", "payment_type", "pickup_community_area", "dropoff_community_area"]].values.tolist()
    # endpoint_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{endpoint}:predict"
    # endpoint_url = get_endpoint_uri(project, region, endpoint.uri)
    endpoint_url = endpoint.uri + ":predict"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {"instances": instances}
    print("请求体:", payload)
    response = requests.post(endpoint_url, headers=headers, json=payload)
    print("Vertex AI endpoint response:", response.text)
    preds = []
    if response.status_code == 200:
        preds = response.json().get("predictions", [])
        print("Predictions:", preds)
    else:
        print("Error:", response.text)


@dsl.pipeline(
    name="vertex-ai-chicago-taxi-e2e-pipeline",
    description="Chicago Taxi E2E pipeline: Dataflow + BigQuery + Vertex AI (sklearn) 全自动化"
)
def chicago_taxi_pipeline(
    project: str,
    region: str,
    temp_location: str,
    dataflow_clean_remote_path: str,  # GCS taxi_dataflow_clean.py 路径
    requirements_file_path: str,     # GCS requirements.txt 路径
    bq_output_table: str,            # Dataflow 输出表
    gcs_model_dir: str,              # GCS 模型存储路径
    model_display_name: str = "chicago-taxi-lr-model",
    endpoint_display_name: str = "chicago-taxi-endpoint",
    gcs_model_path: str = None,
    machine_type: str = "n1-standard-4",
    min_replica_count: int = 1,
    max_replica_count: int = 3,
    dataflow_clean_local_path: str = None, # 占位; 本地 taxi_dataflow_clean.py 路径, 用于上传到 GCS
    gcs_script_dir: str = None,            # 占位; GCS 脚本存储路径
):
    dataflow_task = DataflowPythonJobOp(
        project=project,
        location=region,
        python_module_path=dataflow_clean_remote_path,
        requirements_file_path=requirements_file_path,
        temp_location=temp_location,
        args=[
            "--project", project,
            "--region", region,
            "--temp_location", temp_location,
            "--job_name", f"taxi-dataflow-clean-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}",
            "--save_main_session",
            "--runner", "DataflowRunner",
            "--bq_output_table", bq_output_table,
        ],
    )
    # 等待 Dataflow 任务完成, 
    dataflow_wait = WaitGcpResourcesOp(
        gcp_resources=dataflow_task.outputs["gcp_resources"]
    )
    train_task = train_and_upload_model(
        bq_table=bq_output_table,
        project=project,
        region=region,
        gcs_model_path=gcs_model_path
    ).set_caching_options(False)  # 禁用缓存
    train_task.after(dataflow_wait)
    # 用 importer 生成 UnmanagedContainerModel artifact
    model_artifact = dsl.importer(
        # artifact_uri=gcs_model_path,
        artifact_uri=gcs_model_dir,
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "containerSpec": {
                "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
            }
        }
    )
    model_artifact.after(train_task)
    # 上传模型到 Vertex AI
    upload_task = ModelUploadOp(
        project=project,
        location=region,
        display_name=model_display_name,
        unmanaged_container_model=model_artifact.outputs["artifact"],
    )
    upload_task.after(model_artifact)

    endpoint_task = EndpointCreateOp(
        project=project,
        location=region,
        display_name=endpoint_display_name,
    )
    deploy_task = ModelDeployOp(
        model=upload_task.outputs["model"],
        endpoint=endpoint_task.outputs["endpoint"],
        deployed_model_display_name=model_display_name,
        dedicated_resources_machine_type=machine_type,
        dedicated_resources_min_replica_count=min_replica_count,
        dedicated_resources_max_replica_count=max_replica_count,
        traffic_split={"0": 100},
    )
    deploy_task.after(upload_task, endpoint_task)

    # 通过本地模型推理(远程服务器上)
    predict_task = download_and_predict(
        gcs_model_path=gcs_model_path,
        bq_table=bq_output_table,
        project=project,
        region=region,
    )
    predict_task.after(deploy_task)

    # 通过 Vertex AI endpoint 远程推理(远程服务器上)
    predict_via_endpoint_task = predict_via_endpoint(
        project=project,
        region=region,
        endpoint=endpoint_task.outputs["endpoint"],  # 动态获取
        bq_table=bq_output_table,
    )
    predict_via_endpoint_task.after(deploy_task)



def predict_via_endpoint_local(
    project: str,
    region: str,
    endpoint_id: str,
    bq_table: str,
):
    import pandas as pd
    from google.cloud import bigquery
    import requests
    import google.auth
    import google.auth.transport.requests
    import os
    import google.auth.credentials

    # 获取 access token（如果没传入）
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())
    access_token = credentials.token

    client = bigquery.Client(project=project, location=region)
    df = client.query(f"SELECT * FROM `{bq_table}` LIMIT 5").to_dataframe()
    print("Sample input:\n", df.head())

    instances = df[["trip_seconds", "trip_miles", "payment_type", "pickup_community_area", "dropoff_community_area"]].values.tolist()
    endpoint_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{endpoint_id}:predict"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {"instances": instances}
    print("请求体:", payload)
    response = requests.post(endpoint_url, headers=headers, json=payload)
    print("Vertex AI endpoint response:", response.text)
    preds = []
    if response.status_code == 200:
        preds = response.json().get("predictions", [])
        print("Predictions:", preds)


def upload_file_to_gcs(local_path, gcs_path, project):
    bucket_name = gcs_path.replace('gs://', '').split('/')[0]
    prefix = '/'.join(gcs_path.replace('gs://', '').split('/')[1:])
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(prefix)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to {gcs_path}")

def get_endpoint_uri(project, region, endpoint_id):
    return f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{endpoint_id}:predict"


def run_pipeline(pipeline_params):
    bq_output_table = pipeline_params["bq_output_table"]
    project = pipeline_params["project"]
    region = pipeline_params["region"]

    gcs_script_dir = pipeline_params["gcs_script_dir"]
    gcs_script_path = pipeline_params["dataflow_clean_remote_path"]



    upload_file_to_gcs(pipeline_params["dataflow_clean_local_path"], gcs_script_path, project)
    upload_file_to_gcs("requirements.txt", gcs_script_dir.rstrip("/") + "/requirements.txt", project)

    # 创建 bq dataset
    if '.' in bq_output_table:
        dataset = bq_output_table.split('.')[-2]
        ensure_bq_dataset_exists(project, dataset, region)

    # 编译 pipeline
    compiler.Compiler().compile(
        pipeline_func=chicago_taxi_pipeline,
        package_path='pipeline.yaml'
    )

    # 运行 pipeline
    aiplatform.PipelineJob(
        display_name="chicago-taxi-e2e",
        template_path="pipeline.yaml",
        pipeline_root=get_env("PIPELINE_ROOT", "gs://your-bucket/pipeline_root"),
        parameter_values=pipeline_params
    ).run() 

def call_predict_via_endpoint_local(pipeline_params, endpoint_id='530568236471681024'):
    project = pipeline_params["project"]
    region = pipeline_params["region"]
    bq_output_table = pipeline_params["bq_output_table"]
    predict_via_endpoint_local(
        project=project,
        region=region,
        endpoint_id=endpoint_id,
        bq_table=bq_output_table,
    )

if __name__ == '__main__':
    gcs_script_dir = get_env("GCS_SCRIPT_DIR", required=True)

    pipeline_params = {
        "project": get_env("PROJECT_ID", required=True),
        "region": get_env("REGION", "us-central1"),
        "temp_location": get_env("TEMP_LOCATION", required=True),
        "dataflow_clean_local_path": get_env("DATAFLOW_CLEAN_LOCAL_PATH", "./taxi_dataflow_clean.py"),
        "gcs_script_dir": gcs_script_dir,
        "bq_output_table": get_env("BQ_OUTPUT_TABLE", required=True),
        "gcs_model_dir": get_env("GCS_MODEL_DIR", required=True),
        "gcs_model_path": get_env("GCS_MODEL_PATH", required=True),
        "machine_type": get_env("MACHINE_TYPE", "n1-standard-4"),
        "min_replica_count": get_env("MIN_REPLICA_COUNT", 1),
        "max_replica_count": get_env("MAX_REPLICA_COUNT", 3),
        "endpoint_display_name": get_env("ENDPOINT_DISPLAY_NAME", "chicago-taxi-endpoint"),
        "dataflow_clean_remote_path": gcs_script_dir.rstrip("/") + "/taxi_dataflow_clean.py",
        "requirements_file_path": gcs_script_dir.rstrip("/") + "/requirements.txt",
    }



    run_pipeline(pipeline_params)

    # 在本地调用 Vertex AI endpoint(API)
    call_predict_via_endpoint_local(pipeline_params)
