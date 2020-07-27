
import math
import pandas as pd

from data.DataManager import DataManager as dm


class Preprocessor:
    SEGMENT_MS = 20
    MIN_SEGMENT_LEN = 4

    def process(self):
        """Read the data records and create data features
        """
        accelerometer_data = dm.trip_data_to_df('Pixel_accelerometer')
        segments = self.split_segments(accelerometer_data)
        segment_features = self.features(segments[:2])
        print(segment_features)

    def split_segments(self, df):
        i = 0
        res = []
        while i < len(df):
            temp = df[(df.Time >= df.iloc[i].Time) & (df.Time <
                                                      df.iloc[i].Time + self.SEGMENT_MS)]
            while len(temp) < self.MIN_SEGMENT_LEN:
                self.SEGMENT_MS *= 2
                temp = df[(df.Time >= df.iloc[i].Time) & (df.Time <
                                                          df.iloc[i].Time + self.SEGMENT_MS)]

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

        return df

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
