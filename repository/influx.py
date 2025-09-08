"""
Urea Control System - InfluxDB Repository

This module provides real-time data collection for urea control inference.
The system collects data for exact 5-second windows (1-5s, 6-10s, 11-15s...) 
and returns averaged values for model input.

Key Features:
- Exact 5-second window data collection (1-5s, 6-10s, 11-15s...)
- 1-second interval data within each window (5 data points)
- Data quality validation and NA handling
- Window-averaged values (or last values for specific columns) with aligned timestamps
- Automatic time mapping to window boundaries

Usage Example:
# Get 5-second window data for model inference
current_data = repo.get_current_urea_data([
    "icf_ccs_fg_t_1",     # 소각로 내부 온도
    "icf_scs_fg_t_1",     # 소각로 출구 온도  
    "br1_eo_o2_a",        # 보일러 출구 산소 농도
    "icf_tms_nox_a",      # NOx
    "snr_pmp_uw_s_1"      # 요소수 펌프 Hz
])
# Returns: {"_time_gateway": "2024-01-01 12:00:05", "icf_ccs_fg_t_1": 950.0, "icf_scs_fg_t_1": 920.0, ...}
"""

import datetime
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from influxdb import InfluxDBClient
from influxdb.exceptions import InfluxDBClientError

from utils.config import InfluxConfig
from utils.logger import set_logger

set_logger(__name__)
custom_logger = logging.getLogger(__name__)


class InfluxRepo:
    def __init__(
        self,
        influx_config: Optional[InfluxConfig] = None,
    ):
        """Set influx repo with influx_config

        Parameters
        ----------
        influx_config : Optional[InfluxConfig], optional, by default None
        """
        self.influx_config = influx_config or InfluxConfig()
        self.database_name = self.influx_config.database_name or self.influx_config.source_table_name
        self.source_table_name = self.influx_config.source_table_name
        self.write_table_name = self.influx_config.write_table_name
        
        # InfluxDB 클라이언트 초기화
        self.influx_client = InfluxDBClient(
            host=self.influx_config.host,
            port=self.influx_config.port,
            username=self.influx_config.username,
            password=self.influx_config.password,
            database=self.database_name,
            timeout=self.influx_config.db_timeout,
        )

    def _make_read_query(
        self,
        columns: Union[List[str], Iterable[str]],
        start_time: pd.Timestamp,
        query_range_seconds: int,
        table_name: str,
    ) -> str:
        """Query for read raw data from source table. Note that `start_time` is inclusively
        considered when making the query.
        Steps:
            1. Set range using `query_range_seconds`, considering the `start_time`.
                For example, if `start_time` is 01:59:59 and `query_range_seconds` is 3600,
                the resulting time range would be 01:00:00 ~ 01:59:59 (3600s).
            2. Use the the query string returned from this method.

        Parameters
        ----------
        columns : Union[List[str], Iterable[str]]
        start_time : pd.Timestamp
        query_range_seconds : int
        table_name : str

        Returns
        -------
        str
        """
        end_time = start_time - pd.Timedelta(seconds=query_range_seconds)
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        query = f"""
            SELECT {",".join(columns)}
            FROM {table_name}
            WHERE time > '{end_time_str}' and time <= '{start_time_str}'
        """

        return query

    def read_data(
        self,
        columns: Union[List[str], Iterable[str]],
        start_time: pd.Timestamp,
        query_range_seconds: int,
        table_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read data from provided table.
        This function performs to read data from the specified table.
        Database has already been set by configs and the name of its plant code.
        Database name, and source table name are usually the same. (e.g., "SEGG")

        Args:
            columns (Union[List[str], Iterable[str]]):
                Columns in provided table.
            start_time (pd.Timestamp):
                This value decides start base time of range of data.
            query_range_seconds (int):
                This value decides time range (seconds) of data.
                e.g., `300` means 300 seconds.
            table_name (Optional[str], optional):
                Table name in database.
                When table_name is None then the table will be set as source table.
                Defaults to None.

        Returns:
            pd.DataFrame: Dataframe read from the table.
        """
        table_name = table_name or self.source_table_name
        _q = self._make_read_query(
            columns,
            start_time,
            query_range_seconds,
            table_name,
        )
        try:
            result_set = self.influx_client.query(_q)
            data = pd.DataFrame(result_set.get_points())
            assert not data.empty, "Raw data is empty!"
        except InfluxDBClientError as e:
            custom_logger.error(e)
            return pd.DataFrame()

        return data.drop(columns=["time"]).fillna(np.nan)

    def get_inferred_action(
        self,
        base_time: Union[datetime.datetime, pd.Timestamp, str],
        action_columns: List[str],
        prefix: Optional[str] = "ACT",
    ) -> Dict[str, float]:
        """Get latest inferred action from the write table.

        Args:
            base_time (Union[datetime.datetime, pd.Timestamp, str]):
                Base time to retrieve the latest action.
            action_columns (List[str]):
                Target action columns to be retrieved without prefix.
            prefix (Optional[str], optional):
                Prefix to be prepended on action columns.
                The action columns will be prepended with prefix with separator "_".
                Defaults to None, which is not adding a prefix to the action column.
                Here are some examples:
                    1. prefix="ACT", columns=["col1", "col2"] => columns=["ACT_col1", "ACT_col2"]
                    2. prefix="ACT_", columns=["col1", "col2"] => columns=["ACT_col1", "ACT_col2"]
                    3. prefix=None, columns=["col1", "col2"] => columns=["col1", "col2"]

        Returns:
            Dict[str, float]: Latest action dict.
        """
        if prefix and not prefix.endswith("_"):
            prefix += "_"
        else:
            prefix = ""

        action_queries = [f"{prefix + action_name} AS {action_name}" for action_name in action_columns]
        _q = f"""
            SELECT {",".join(action_queries)}
            FROM {self.write_table_name}
            WHERE time <= '{base_time.strftime("%Y-%m-%d %H:%M:%S")}'
            ORDER BY time DESC LIMIT 1
        """

        retry_count = 0
        max_retries = self.influx_config.last_wf_action_max_retries
        inference_tolerance_period = pd.Timedelta(self.influx_config.last_wf_action_tolerance)
        retry_period = pd.Timedelta(self.influx_config.last_wf_action_retry_period)

        while retry_count <= max_retries:
            if retry_count != 0:
                custom_logger.info(f"Retrying to get the latest action ({retry_count}/{max_retries})")

            lastest_action_dict = list(self.influx_client.query(_q).get_points())[0]
            assert lastest_action_dict, (
                "Latest action dict is empty! "
                f"{[prefix + action_name for action_name in action_columns]} "
                f"do not exist in the table '{self.write_table_name}'."
            )

            base_time: pd.Timestamp = pd.to_datetime(base_time, utc=True)
            inferred_time: pd.Timestamp = pd.to_datetime(lastest_action_dict.pop("time"), utc=True)
            if abs(inferred_time - base_time) <= inference_tolerance_period:
                return lastest_action_dict

            retry_count += 1
            time.sleep(retry_period.total_seconds())

        custom_logger.warn(
            f"Fail to get recent action with base time of {str(base_time)}, and "
            f"finally got {str(inferred_time)} as the latest inference time. "
            "The latest action is retrieved from actual data, not from the inferenced data."
        )
        lastest_action_dict = self.get_actual_action(base_time, action_columns)
        return lastest_action_dict

    def get_actual_action(
        self,
        base_time: Union[datetime.datetime, pd.Timestamp, str],
        action_columns: List[str],
    ) -> Dict[str, float]:
        """Get latest actual action from the read table.
        Args:
        base_time (Union[datetime.datetime, pd.Timestamp, str]):
            Base time to retrieve the latest action.
        action_columns (List[str]):
            Target action columns to be retrieved without prefix.

        Returns:
            Dict[str, float]: Latest action dict.
        """

        _q = f"""
            SELECT {",".join(action_columns)}
            FROM {self.source_table_name}
            WHERE time <= '{base_time.strftime("%Y-%m-%d %H:%M:%S")}'
            ORDER BY time DESC LIMIT 1
        """
        lastest_action_dict = list(self.influx_client.query(_q).get_points())[0]
        lastest_action_dict.pop("time", None)  # Remove unwanted key
        return lastest_action_dict

    def get_custom_action_code(
        self,
        base_time: Union[datetime.datetime, pd.Timestamp, str],
        custom_action_code_column: str,
    ) -> int:
        _q = f"""
            SELECT {custom_action_code_column}
            FROM {self.write_table_name}
            WHERE time <= '{base_time.strftime("%Y-%m-%d %H:%M:%S")}'
            ORDER BY time DESC LIMIT 1
        """

        last_custom_action_code = list(self.influx_client.query(_q).get_points())[0]
        last_custom_action_code.pop("time", None)  # Remove unwanted key
        if custom_action_code_column not in last_custom_action_code:
            # If this is the first run, the column may not exist.
            custom_logger.warn(f"'{custom_action_code_column}' column is not found! Set as 0.")
            return 0

        return last_custom_action_code[custom_action_code_column]

    def convert_data_to_influx_format(
        self,
        result_df: pd.DataFrame,
        monitor_features: Optional[List[str]] = None,
        prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert result pd.DataFrame to Influx json format.

        Args:
            result_df (pd.DataFrame):
                What if simulation results, predicted action or observation results as DataFrame object.
                It should have the followings:
                    - DatetimeIndex without microseconds (Smallest time unit must be second.)
                    - Columns which will be written into the table.
                    - For example, returned object of `ModelSVC.dm_pred`.
            monitor_features (Optional[List[str]], optional):
                Monitoring target features to save and manage.
                If None, all columns are regarded to be written. Defaults to None.
            prefix (Optional[str], optional):
                Prefix to be prepended on the feature name.
                The name of features will be combined with prefix with separator "_".
                Defaults to None, which is not adding a prefix to feature name.
                Here are some examples:
                    1. If prefix="ACT", columns=["col1", "col2"] => columns=["ACT_col1", "ACT_col2"]
                    2. If prefix="ACT_", columns=["col1", "col2"] => columns=["ACT_col1", "ACT_col2"]
                    3. If prefix=None, columns=["col1", "col2"] => columns=["col1", "col2"]

        Returns:
            Dict[str, Any]:
                Converted results with InfluxDB input format.
                Each dictionary is organized in-line as InfluxDB input structure.
                e.g., InfluxDB input structure:
                {
                    "time" : "2023-01-01 00:00:00", (microseconds must not exist here)
                    "fields": {
                        "YOUR_FIELD_NAME": "VALUE"
                        ...
                    }
                }

        Reference
            https://influxdb-python.readthedocs.io/en/latest/include-readme.html#examples
        """
        if monitor_features:
            result_df = result_df[result_df.columns.intersection(monitor_features)]
        if prefix:
            if not prefix.endswith("_"):
                prefix += "_"
            result_df = result_df.add_prefix(prefix)

        result_df.index.name = "_time_gateway"
        # Convert the datetime index to string type for writing into InfluxDB
        result_df.index = result_df.index.astype("str")
        result_df = result_df.add_prefix(prefix)

        records = (
            [  # To make InfluxDB protocol format
                {
                    "time": str(timestamp),
                    "fields": fields,
                }
                for timestamp, fields in result_df.to_dict(orient="index").items()
            ],
        )

        return records

    def write_to_influx(
        self,
        data: List[Dict[str, Dict[str, float]]],
        table_name: Optional[str] = None,
    ) -> None:
        """Write data to the provided table.

        Args:
            data (List[Dict[str, Dict[str, float]]]):
                Data which will be written into the write table.
                Structure of each element must follow InfluxDB input format as follows:
                e.g.,
                    {
                        "time" : "2023-01-01 00:00:00", (microseconds must not exist here)
                        "fields": {
                            "YOUR_FIELD_NAME": "VALUE"
                        }
                    }
            table_name (Optional[str], optional):
                The name of the table to write the data.
                When table does not exist, new table (measurement in InfluxDB) will be created.
                When `table_name` is None, it will be set as `InfluxConfig.write_table_name`.
                Defaults to None, which is `InfluxConfig.write_table_name`.
        """
        if table_name is None:
            table_name = self.write_table_name
        for idx in range(len(data)):
            data[idx]["measurement"] = table_name
        self.influx_client.write_points(
            points=data,
            protocol="json",
        )


