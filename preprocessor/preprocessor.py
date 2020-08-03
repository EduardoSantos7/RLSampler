
import math
import pandas as pd

from math import ceil
from concurrent.futures import ProcessPoolExecutor

from data.DataManager import DataManager as dm


class Preprocessor:
    SEGMENT_MS = 20
    MIN_SEGMENT_LEN = 4
    AVG_FEATURES_INTERVALS = 60000

    def process(self, workers=8):
        """Read the data records and create data features
        """
        accelerometer_data = dm.trip_data_to_df('Pixel_gyro_1_042317_1126')
        segments = self.split_segments(accelerometer_data)

        dim = ceil(len(segments) / workers)
        chunks = (segments[k: k + dim] for k in range(0, len(segments), dim))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self.features, chunk) for chunk in chunks
            ]

        segment_features = []

        for future in futures:
            segment_features.append(future.result())

        segments_df = pd.concat(segment_features, ignore_index=True)
        segments_df = segments_df.sort_values('Time')

        # Average features to 1 minute intervals
        segments = self.split_segments(
            segments_df, time_intervals=self.AVG_FEATURES_INTERVALS)

        dim = ceil(len(segments) / workers)
        chunks = (segments[k: k + dim] for k in range(0, len(segments), dim))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self.average_features, chunk) for chunk in chunks
            ]

        average_features = []
        for future in futures:
            average_features.append(future.result())

        avg_features_df = pd.concat(average_features, ignore_index=True)
        avg_features_df = avg_features_df.sort_values('Time')
        print(avg_features_df)

    def split_segments(self, df, time_intervals=None):
        if not time_intervals:
            time_intervals = self.SEGMENT_MS

        i = 0
        res = []
        while i < len(df):
            temp = df[(df.Time >= df.iloc[i].Time) & (df.Time <
                                                      df.iloc[i].Time + time_intervals)]
            while len(temp) < self.MIN_SEGMENT_LEN:
                time_intervals *= 2
                temp = df[(df.Time >= df.iloc[i].Time) & (df.Time <
                                                          df.iloc[i].Time + time_intervals)]

            i += len(temp)
            res.append(temp)

        return res

    def features(self, segments):
        segment_features = []

        for segment in segments:
            msm = self.mean_square_magnitude(segment)
            variance = self.variance_magnitude(segment, msm)
            df = pd.DataFrame()
            df['Time'] = segment['Time']
            df['msm'] = msm
            df['variance'] = variance
            segment_features.append(df)

        return pd.concat(segment_features)

    def mean_square_magnitude(self, segment):

        # Calcuate the pow(2) in each elemnt that's not a date
        squares = segment.loc[:, [
            col for col in segment.columns if col != 'Time']].pow(2)

        # Sum all the elements
        sums = sum(squares.sum())

        return sums / len(segment)

    def variance_magnitude(self, segment, msm):
        squares = segment.loc[:, [
            col for col in segment.columns if col != 'Time']].pow(2)

        row_sums = [math.sqrt(squares.iloc[row].sum())
                    for row in range(len(squares))]

        sums = 0
        for num in row_sums:
            temp = num - msm
            temp *= temp  # Square
            sums += temp

        return math.sqrt(sums / len(segment))

    def average_features(self, segments):
        segment_features = []

        for segment in segments:
            data = {
                'Time': segment['Time'].iloc[-1],
                'msm': segment['msm'].mean(),
                'variance': segment['variance'].mean(),
            }

            segment_features.append(data)

        return pd.DataFrame(segment_features)
