import os.path
import pandas as pd
import yaml
import boto3
from io import StringIO
import cv2
import numpy as np


def load_config_from_yaml(config_file_path):
    with open(config_file_path) as file:
        config = yaml.safe_load(file)
    return config


def load_label_csv(csv_file_path, config):
    if os.path.isfile(csv_file_path):
        label_data = pd.read_csv(csv_file_path, header=0, index_col=False)
    else:
        s3 = boto3.client("s3")
        csv_obj = s3.get_object(Bucket=config['bucket'], Key=csv_file_path)
        body = csv_obj['Body']
        csv_string = body.read().decode('utf-8')
        label_data = pd.read_csv(StringIO(csv_string), header=0, index_col=False)
    return label_data


def load_cv2_image_from_s3(image_path, config):
    s3 = boto3.client('s3')
    return cv2.imdecode(np.frombuffer(s3.get_object(Bucket=config['bucket'], Key=image_path)['Body'].read(), np.uint8),
                        cv2.IMREAD_COLOR)


def load_labels_from_df(df, image_path):
    return (df[df['image_path'] == image_path]['holes'].values[0],
            df[df['image_path'] == image_path]['growth'].values[0])
