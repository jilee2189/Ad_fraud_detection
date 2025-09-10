import sys
import os 
import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from adfrauddetection.constant.training_pipeline import TARGET_COLUMN
from typing import List, Optional

from adfrauddetection.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
from sklearn.model_selection import train_test_split
from adfrauddetection.entity.config_entity import DataTransformationConfig
from adfrauddetection.exception.exception import AdfrauddetectionException
from adfrauddetection.logging.logger import logging
from adfrauddetection.utils.main_utils.major_utils import save_numpy_array_data, save_object

class DataTransformation: 
    def __init__(self, data_ingestion_artifact :DataIngestionArtifact, 
                 data_transformation_config: DataTransformationConfig): 
        try: 
            self.data_ingestion_artifact: DataIngestionArtifact= data_ingestion_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e: 
            raise AdfrauddetectionException(e,sys)
    @staticmethod 
    def read_data(file_path) -> pd.DataFrame: 
        try: 
            return pd.read_csv(file_path)
        except Exception as e:  
            raise AdfrauddetectionException(e, sys)
    @staticmethod
    def add_count(
        df: pd.DataFrame,
        by: list[str],
        name: str,
        dtype: str = "uint32",
        fillna_value: int = 0
    ) -> pd.DataFrame:
        """
        Count rows per group, including groups with NaN in keys.
        """
        # include NaN as a valid key to avoid missing rows
        grp = df.groupby(by, dropna=False, sort=False)
        df[name] = grp.transform("size")

        # fill any residual NaNs defensively and cast
        # (shouldn't happen with dropna=False, but keeps things safe if you later map from train->test)
        df[name] = df[name].fillna(fillna_value).astype(dtype)
        return df
    
    @staticmethod
    def add_nunique(
        df: pd.DataFrame,
        by: List[str],
        col: str,
        name: str,
        dtype: str = "uint32"
    ) -> pd.DataFrame:
        """
        Add a column with the number of UNIQUE values of `col` per group.

        Example:
            add_nunique(trainset, by=["ip"], col="app", name="ip_unique_apps")
            -> for each row, how many distinct apps that ip has clicked

        Parameters
        ----------
        df : DataFrame
            Input dataframe (modified in place and also returned).
        by : list of str
            Column(s) to group by.
        col : str
            Column for which to count unique values.
        name : str
            Name of the new unique-count column.
        dtype : str
            Integer dtype to cast to.

        Returns
        -------
        DataFrame
            Same df with a new column `name`.
        """
        df[name] = df.groupby(by)[col].transform("nunique").astype(dtype)
        return df
    
    @staticmethod
    def add_cumcount(
        df: pd.DataFrame,
        by: List[str],
        col: str,
        name: str,
        dtype: str = "uint32",
        sort_by: Optional[str] = None,
        ascending: bool = True
    ) -> pd.DataFrame:
        """
        Add a column with the CUMULATIVE COUNT (0,1,2,...) of `col` within each group.

        Example:
            add_cumcount(trainset, by=["ip", "device", "os"], col="app", name="seq_click_idx")
            -> within each (ip, device, os) group, label each row with its running index

        Tip:
            If you want cumcount to reflect time order, pass `sort_by="click_time"`
            (or another timestamp column). This function will temporarily sort and
            then restore the original order.

        Parameters
        ----------
        df : DataFrame
            Input dataframe (modified in place and also returned).
        by : list of str
            Column(s) to group by.
        col : str
            Column used only to anchor the group for cumcount.
            (cumcount ignores the values and just numbers rows in each group)
        name : str
            Name of the new cumulative-count column.
        dtype : str
            Integer dtype to cast to.
        sort_by : Optional[str]
            Column to sort by BEFORE computing cumcount, e.g., a timestamp.
            If None, use the current row order.
        ascending : bool
            Sort ascending if sort_by is provided.

        Returns
        -------
        DataFrame
            Same df with a new column `name`.
        """
        if sort_by is None:
            # No sorting: use current order
            df[name] = df.groupby(by)[col].cumcount().astype(dtype)
            return df
        else:
            # Sort, compute, then restore original order
            original_index = df.index
            df_sorted = df.sort_values(sort_by, ascending=ascending).copy()
            df_sorted[name] = df_sorted.groupby(by)[col].cumcount().astype(dtype)
            # Put values back in original order
            df.loc[df_sorted.index, name] = df_sorted[name]
            # Ensure dtype
            df[name] = df[name].astype(dtype)
            df = df.loc[original_index]
            return df

    @staticmethod
    def add_group_mean(df, by, col, name, past_only=False, time_col=None, dtype="float32"):
        if past_only:
            assert time_col is not None, "past_only=True requires time_col"
            df = df.sort_values(time_col)
            g = df.groupby(by)[col].apply(lambda s: s.shift().expanding().mean())
            df[name] = g.values.astype(dtype)
            df = df.sort_index()
        else:
            df[name] = df.groupby(by)[col].transform("mean").astype(dtype)
        return df

    @staticmethod
    def add_group_var(df, by, col, name, past_only=False, time_col=None, dtype="float32"):
        s = pd.to_numeric(df[col], errors="coerce")
        if past_only:
            assert time_col is not None, "past_only=True requires time_col"
            df_sorted = df.sort_values(time_col).copy()
            g = df_sorted.groupby(by)[col].apply(lambda x: x.shift().expanding().var(ddof=0))
            df[name] = g.values.astype(dtype)
            df.sort_index(inplace=True)
        else:
            # ddof=0 makes 1-row groups → variance 0 (not NaN)
            df[name] = df.groupby(by)[col].transform(lambda x: x.var(ddof=0)).astype(dtype)
        return df

    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try: 
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_ingestion_artifact.trained_file_path)
            #test_df = DataTransformation.read_data(self.data_ingestion_artifact.test_file_path) 
            ## transformation of train_df
            train_df = train_df.drop('attributed_time', axis=1,errors='ignore')
            train_df['click_time'] = pd.to_datetime(train_df['click_time'])
            train_df['year'] = train_df.click_time.dt.year
            train_df['month'] = train_df.click_time.dt.month
            train_df['day'] = train_df.click_time.dt.day
            train_df['hour'] = train_df.click_time.dt.hour
            train_df['min'] = train_df.click_time.dt.minute
            train_df['sec'] = train_df.click_time.dt.second
            self.add_count(train_df, by=["ip","day","hour"], name="ip_tcount")
            self.add_count(train_df, by=["ip","app"],        name="ip_app_count")
            self.add_count(train_df, by=["ip","app","os"],   name="ip_app_os_count")
            self.add_nunique(train_df, by=["ip"],            col="device",  name="X5")
            self.add_nunique(train_df, by=["app"],           col="channel", name="X6")
            self.add_nunique(train_df, by=["ip","device","os"], col="app",  name="X8")
            self.add_cumcount(train_df, by=["ip", "device", "os"], col="app", name="X1")
            self.add_group_var(train_df,  by=['ip','day','channel'], col='hour', name='ip_tchan_count')
            self.add_group_var(train_df,  by=['ip','app','os'],      col='hour', name='ip_app_os_var')
            self.add_group_var(train_df,  by=['ip','app','channel'], col='day',  name='ip_app_channel_var_day')
            self.add_group_mean(train_df, by=['ip','app','channel'], col='hour', name='ip_app_channel_mean_hour')
            train_df = train_df.drop('click_time', axis=1)
            new_train_df = train_df[train_df['day'].isin([6, 7, 8])]
            new_test_df  = train_df[train_df['day'].isin([9])]

            ## training dataframe 
            input_feature_train_df = new_train_df.drop(columns = [TARGET_COLUMN], axis=1)
            target_feature_train_df = new_train_df[TARGET_COLUMN]
            # training numpy arrays

            train_arr = np.c_[input_feature_train_df.values,
                            target_feature_train_df.values.reshape(-1, 1)]
            
            ## testing dataframe 
            input_feature_test_df = new_test_df.drop(columns = [TARGET_COLUMN], axis=1)
            target_feature_test_df = new_test_df[TARGET_COLUMN]
            
            # testing numpy arrays
            test_arr = np.c_[input_feature_test_df.values,
                            target_feature_test_df.values.reshape(-1, 1)]

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path, array=test_arr, )
            data_transformation_artifact=DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
        except Exception as e: 
            raise AdfrauddetectionException(e, sys)