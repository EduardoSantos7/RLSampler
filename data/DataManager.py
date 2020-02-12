from os import listdir
from difflib import get_close_matches

import pandas as pd

TRIP_PATH = './Trip'
PREDICTION_PSEUDONAME = 'Pixel_activity_'
ACCELEROMETER_PSEUDONAME = 'Pixel_accelerometer'


class DataManager:

    def trip_data_to_df(self, data_psudoname, trip=1):

        file = self._file(data_psudoname, trip)
        data_path = f'{TRIP_PATH} {trip}/{file}'
        df = pd.read_csv(data_path)

        return df

    def _file(self, pseudoname, trip):
        trip_path = f'{TRIP_PATH} {trip}'
        files = listdir(trip_path)
        # Unpack the close match
        [file] = get_close_matches(pseudoname, files)
        return file
