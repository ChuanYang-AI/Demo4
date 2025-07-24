#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhipeng
# @Email: zhangzhipeng@polymericcloud.com
# @Date:   2025-05-14 14:38:15
# @Last Modified By:    zhipeng
# @Last Modified: 2025-05-20 19:34:02


import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import os
import time
from google.cloud import bigquery
import pandas as pd
import logging

def get_env(key, default=None, required=False):
    v = os.environ.get(key, default)
    if required and v is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return v

def filter_and_clean(row):
    try:
        # 1. trip_seconds > 60 and < 6*60*60
        if row['trip_seconds'] is None or not (60 < row['trip_seconds'] < 6*60*60):
            return False
        # 2. trip_miles > 0 and < 300
        if row['trip_miles'] is None or not (0 < row['trip_miles'] < 300):
            return False
        # 3. trip_total > 3 and < 3000
        if row['trip_total'] is None or not (3 < row['trip_total'] < 3000):
            return False
        # 4. pickup_community_area, dropoff_community_area not null
        if row['pickup_community_area'] is None or row['dropoff_community_area'] is None:
            return False
        # 5. payment_type in ["Credit Card", "Cash"]
        if row['payment_type'] not in ("Credit Card", "Cash"):
            return False
        # 6. trip_hours = round(trip_seconds / 3600, 2)
        trip_hours = round(row['trip_seconds'] / 3600, 2)
        if trip_hours > 2:
            return False
        # 7. trip_speed = round(trip_miles / trip_hours, 2)
        if trip_hours == 0:
            return False
        trip_speed = round(row['trip_miles'] / trip_hours, 2)
        if trip_speed > 70:
            return False
        return True
    except Exception:
        return False

def clean_row(row):
    # trip_hours, trip_speed
    trip_hours = round(row['trip_seconds'] / 3600, 2)
    trip_speed = round(row['trip_miles'] / trip_hours, 2) if trip_hours > 0 else None
    # payment_type 编码
    payment_type = row.get('payment_type')
    payment_type = 0 if payment_type == 'Credit Card' else 1
    # 时间特征
    ts = pd.to_datetime(row['trip_start_timestamp'])
    dayofweek = ts.dayofweek
    hour = ts.hour
    # dayofweek: 0=周末, 1=工作日
    dayofweek = 0 if dayofweek in [5, 6] else 1
    # hour: 0=夜间, 1=白天
    hour = 0 if hour in [23, 0, 1, 2, 3, 4, 5, 6, 7] else 1
    return {
        'taxi_id': row.get('taxi_id'),
        'trip_start_timestamp': row.get('trip_start_timestamp'),
        'trip_seconds': row.get('trip_seconds'),
        'trip_miles': row.get('trip_miles'),
        'trip_total': row.get('trip_total'),
        'payment_type': payment_type,
        'pickup_community_area': row.get('pickup_community_area'),
        'dropoff_community_area': row.get('dropoff_community_area'),
        'trip_hours': trip_hours,
        'trip_speed': trip_speed,
        'dayofweek': dayofweek,
        'hour': hour,
        'company': row.get('company')  # 可选保留
    }


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bq_output_table', required=True)
    # pipeline 参数不要定义, 否则会被默认 args 获取.
    # parser.add_argument('--project', required=True)
    # parser.add_argument('--region', required=True)
    # parser.add_argument('--temp_location', required=True)
    # parser.add_argument('--staging_location', required=False)

    args, pipeline_args = parser.parse_known_args()

    # 生成唯一 job_name，防止重名
    default_job_name = f"taxi-dataflow-clean-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"

    from apache_beam.options.pipeline_options import PipelineOptions
    options = PipelineOptions(pipeline_args)
    all_opts = options.get_all_options()
    project = all_opts.get('project', None)
    region = all_opts.get('region', None)
    temp_location = all_opts.get('temp_location', None)
    staging_location = all_opts.get('staging_location', None) or temp_location
    job_name = all_opts.get('job_name', None) or default_job_name
    runner = all_opts.get('runner', None)
    save_main_session = all_opts.get('save_main_session', True)

    print("all_opts:", all_opts)
    print("project:", project)
    print("region:", region)
    print("temp_location:", temp_location)
    print("staging_location:", staging_location)
    print("bq_output_table:", args.bq_output_table)
    print("runner:", runner)
    print("save_main_session:", save_main_session)
    print("job_name:", job_name)
    # options.view_as(PipelineOptions).job_name = job_name
    # options.view_as(PipelineOptions).save_main_session = True
    # options.view_as(PipelineOptions).runner = 'DataflowRunner'

    # 不用 with，手动构建 pipeline
    p = beam.Pipeline(options=options)
    (
        p
        | 'ReadFromBQ' >> beam.io.ReadFromBigQuery(
            query="""
            SELECT taxi_id, trip_start_timestamp, trip_seconds, trip_miles, trip_total, payment_type, pickup_community_area, dropoff_community_area, company
            FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
            WHERE trip_start_timestamp >= '2018-05-12' AND trip_end_timestamp <= '2018-06-18'
            """,
            use_standard_sql=True
        )
        | 'FilterValid' >> beam.Filter(filter_and_clean)
        | 'CleanRow' >> beam.Map(clean_row)
        | 'WriteToBQ' >> beam.io.WriteToBigQuery(
            args.bq_output_table,
            schema='taxi_id:STRING,trip_start_timestamp:TIMESTAMP,trip_seconds:INTEGER,trip_miles:FLOAT,trip_total:FLOAT,payment_type:INTEGER,pickup_community_area:INTEGER,dropoff_community_area:INTEGER,trip_hours:FLOAT,trip_speed:FLOAT,dayofweek:INTEGER,hour:INTEGER,company:STRING',
            write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE
        )
    )
    result = p.run()
    print('Dataflow job running..., job:', dir(result))
    if hasattr(result, 'id'):
        print('Dataflow job id:', result.id)
    print(f"Dataflow job {job_name} running..., result: {result}, result type: {type(result)}")
    result.wait_until_finish()
    print(f"Dataflow job {job_name} finished with state {result.state}.. result: {result}, result type: {type(result)}")
    return result


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # 设置日志级别, 非常重要, 必须设置!
    # https://github.com/apache/beam/issues/35013
    # https://github.com/apache/beam/pull/34952
    # INFO:apache_beam.runners.dataflow.internal.apiclient:To access the Dataflow monitoring console, please navigate to https://console.cloud.google.com/dataflow/jobs/us-central1/2025-05-14_20_54_31-10431947524306901546?project=cy-aispeci-demo
    logging.getLogger('apache_beam.runners.dataflow.internal.apiclient').setLevel(logging.INFO)
    run()
