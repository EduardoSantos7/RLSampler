from os import listdir, getcwd
from difflib import get_close_matches

import pandas as pd

TRIP_PATH = '/data/Trip'
PREDICTION_PSEUDONAME = 'Pixel_activity_'
ACCELEROMETER_PSEUDONAME = 'Pixel_accelerometer'


class DataManager:

    @classmethod
    def trip_data_to_df(cls, data_psudoname, trip=1):

        file = cls._file(data_psudoname, trip)
        data_path = f'{getcwd()}{TRIP_PATH} {trip}/{file}'
        df_data = pd.read_csv(data_path, header=1)
        return df_data

        # file = cls._file(PREDICTION_PSEUDONAME, trip)
        # data_path = f'{getcwd()}{TRIP_PATH} {trip}/{file}'
        # df_result = pd.read_csv(data_path, header=1)

        # return df_data.merge(df_result, on="Time", how='left').drop_duplicates()

    @staticmethod
    def _file(pseudoname, trip):
        trip_path = f'{getcwd()}{TRIP_PATH} {trip}'
        files = listdir(trip_path)
        for file in files:
            if pseudoname in file:
                return file
        # Unpack the close match
        # file = get_close_matches(pseudoname, files)[0]
        return file
