
import math
import pandas as pd

from math import ceil
from concurrent.futures import ProcessPoolExecutor

from data.DataManager import DataManager as dm


class Preprocessor:
    SEGMENT_MS = 20
    MIN_SEGMENT_LEN = 4
    AVG_FEATURES_INTERVALS = 60000

    def process(self):
        trips = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        workers = 4
        dim = ceil(len(trips) / workers)
        chunks = (trips[k: k + dim] for k in range(0, len(trips), dim))
        trips_df = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self.get_joined_trips_data, chunk) for chunk in chunks
            ]

        for future in futures:
            trips_df.extend(future.result())

        complete = pd.concat(trips_df)
        complete.to_csv(f'data/{len(trips)}_trips.csv')

    def get_joined_trips_data(self, trips):

        joineds = []

        for trip in trips:
            files = ['Pixel_accelerometer', f'Pixel_gyro_{trip}']
            results = []

            for file in files:
                result = self.feature_processing(file, trip=trip)
                file = self.clean_file_name(file)
                result.rename(columns={'msm': f'msm_{file}',
                                       'variance': f'variance_{file}'}, inplace=True)
                results.append(result)

            # results = []
            # for future in futures:
                # results.append(future.result())

            pre_result = pd.concat(results, axis=1)
            result = self.remove_dup_columns(pre_result)
            # print(result)

            data = dm.trip_data_to_df("Pixel_activity", trip=trip)
            # Average features to 1 minute intervals
            data_segments = self.split_segments(
                data, time_intervals=self.AVG_FEATURES_INTERVALS)

            avg_classification_df = self.average_classification(data_segments)

            joined_df = self.join_sensors_classification_data(
                result, avg_classification_df)

            joineds.append(joined_df)

        return joineds

    def feature_processing(self, filename, trip=1, workers=8):
        """Read the data records and create data features
        """
        accelerometer_data = dm.trip_data_to_df(filename, trip=trip)
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

        return avg_features_df

    def split_segments(self, df, time_intervals=None):
        if not time_intervals:
            time_intervals = self.SEGMENT_MS

        i = 0
        res = []
        while i < len(df) - 1:
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

    def remove_dup_columns(self, frame):
        keep_names = set()
        keep_icols = list()
        for icol, name in enumerate(frame.columns):
            if name not in keep_names:
                keep_names.add(name)
                keep_icols.append(icol)
        return frame.iloc[:, keep_icols]

    def average_classification(self, segments):
        segments_classification = []

        for segment in segments:
            segment_len = len(segment)
            data = {
                # Time will be the mid time in the segment
                'Time': segment['Time'].iloc[segment_len // 2],
                'PROBABLE': self.most_frequent_classification(segment)
            }

            segments_classification.append(data)

        return pd.DataFrame(segments_classification)

    def most_frequent_classification(self, segment):
        counter = {}
        for label in segment['PROBABLE']:
            counter[label] = counter.get(label, 0) + 1

        max_appears = max(counter.values())

        for label, count in counter.items():
            if count == max_appears:
                return label

    def join_sensors_classification_data(self, sensors_df, class_df, drop_na_classes=True):
        class__assigned = []
        for sensor_row in range(len(sensors_df)):
            class__assigned.append(self.get_class_assigned(
                sensors_df.iloc[sensor_row]['Time'], class_df))

        joined_df = sensors_df.copy(deep=True)
        joined_df['PROBABLE'] = class__assigned

        if drop_na_classes:
            joined_df.dropna(subset=['PROBABLE'], inplace=True)

        return joined_df

    def get_class_assigned(self, time, class_df):
        half_avg_features_time = self.AVG_FEATURES_INTERVALS // 2
        time_range = class_df[(class_df.Time >= time - half_avg_features_time) &
                              (class_df.Time < time + half_avg_features_time)]

        class_assigned = None
        current_min_range_time = float('inf')

        for row in range(len(time_range)):
            time_diff = abs(time_range.iloc[row]['Time'] - time)
            if time_diff < current_min_range_time:
                current_min_range_time = time_diff
                class_assigned = time_range.iloc[row]['PROBABLE']

        return class_assigned

    def clean_file_name(self, filename):
        NAMES = [
            'accelerometer', 'gravity', 'gyro', 'linacc', 'location', 'magneticField', 'rotation'
        ]
        for correct_name in NAMES:
            if correct_name in filename:
                return correct_name
        return ""
