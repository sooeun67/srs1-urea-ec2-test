import json
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd

from utils.config import KDSConfig


class KDSRepo:
    def __init__(self, aws_config: Optional[KDSConfig] = None) -> None:
        """initialize KDS Repo with given aws config for urea control system

        Parameters
        ----------
        aws_config : Optional[KDSConfig], optional, by default None
        """
        config = aws_config or KDSConfig()
        self.region_name = config.region_name
        self.stream_name = config.stream_name
        self.kinesis_client = boto3.client(
            "kinesis",
            region_name=self.region_name,
        )

    def convert_urea_control_data_to_kds_format(
        self,
        result_df: pd.DataFrame,
        monitor_features: Optional[List[str]] = None,
        prefix: Optional[str] = "UREA",
    ) -> List[Dict[str, Any]]:
        """Convert urea control pd.DataFrame to KDS json format.

        Args:
            result_df (pd.DataFrame):
                Urea control results as DataFrame object.
                It should have the followings:
                    - DatetimeIndex without microseconds (Smallest time unit must be second.)
                    - Columns which will be inserted into stream.
                    - For example, predicted NOx, optimized pump Hz, control status, etc.
            monitor_features (Optional[List[str]], optional):
                Monitoring target features to save and manage.
                If None, all columns are regarded to be written. Defaults to None.
            prefix (Optional[str], optional):
                Prefix to be prepended on the feature name.
                The name of features will be combined with prefix with separator "_".
                Defaults to "UREA", which adds "UREA_" prefix to feature name.
                Here are some examples:
                    1. If prefix="UREA", columns=["hz", "nox_pred"] => columns=["UREA_hz", "UREA_nox_pred"]
                    2. If prefix="UREA_", columns=["hz", "nox_pred"] => columns=["UREA_hz", "UREA_nox_pred"]
                    3. If prefix=None, columns=["hz", "nox_pred"] => columns=["hz", "nox_pred"]

        Returns:
            List[Dict[str, Any]]:
                Converted results with constant additional informations.
                Additional Informations:
                    _data_source: "lambda"
                    _schema_version: "0.1"
                    _model_version: "20241201" (urea control model version)
                    _model_type: "urea_gp" (Gaussian Process model)
        """
        if monitor_features:
            result_df = result_df[result_df.columns.intersection(monitor_features)]
        if prefix:
            if not prefix.endswith("_"):
                prefix += "_"
            result_df = result_df.add_prefix(prefix)

        result_df.index.name = "_time_gateway"
        # Convert the datetime index to string type for KDS stream insertion
        result_df.index = pd.to_datetime(result_df.index).strftime("%Y-%m-%d %H:%M:%S")
        result_df = result_df.assign(
            _data_source="lambda",
            _schema_version="0.1",
            _model_version="20241201",  # Urea control model version
            _model_type="urea_gp",      # Gaussian Process model type
        )
        records = result_df.reset_index().to_dict("records")

        return records

    def write_urea_control_data_to_kds(
        self,
        payloads: List[Dict[str, Any]],
        model_name: str,
        stream_name: Optional[str] = None,
    ) -> None:
        """Write urea control payloads (JSON) to AWS Kinesis Data Stream.

        Args:
            payloads (List[Dict[str, Any]]):
                Urea control data (predicted NOx, optimized pump Hz, control status, etc.).
                Each payload must be datetime string with seconds format (without microseconds)
                through "_time_gateway" key.
                e.g.,
                {
                    "_time_gateway" : "2023-01-01 00:00:00", (microseconds must not exist here)
                    "UREA_hz" : 50.0,
                    "UREA_nox_pred" : 80.0,
                    "UREA_control_status" : 1,
                    "_model_type" : "urea_gp",
                    ...
                }
            model_name (str):
                Model name that decide partition key prefix (e.g., "urea_gp_model").
            stream_name (Optional[str], optional):
                Kinesis Stream name generally got from OS environment.
                When value is None, it will be set from OS environment.
        """
        for payload in payloads:
            partition_key = model_name + payload["_time_gateway"]

            self.kinesis_client.put_record(
                StreamName=stream_name or self.stream_name,
                Data=json.dumps(payload),
                PartitionKey=partition_key,
            )
