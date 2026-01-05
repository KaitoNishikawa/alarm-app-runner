import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants
from source.preprocessing.epoch import Epoch
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService


class HeartRateFeatureService(object):
    WINDOW_SIZE = 10 * 30 - 15

    @staticmethod
    def load(subject_id):
        heart_rate_feature_path = HeartRateFeatureService.get_path(subject_id)
        feature = np.load(str(heart_rate_feature_path))
        return feature

    @staticmethod
    def get_path(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_hr_feature.npy')

    @staticmethod
    def write(subject_id, feature):
        heart_rate_feature_path = HeartRateFeatureService.get_path(subject_id)
        np.save(heart_rate_feature_path, feature)

    @staticmethod
    def get_path_for_mean_raw(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_hr_mean_raw_feature.npy')

    @staticmethod
    def write_mean_raw(subject_id, feature):
        mean_feature_path = HeartRateFeatureService.get_path_for_mean_raw(subject_id)
        np.save(mean_feature_path, feature)

    @staticmethod
    def get_path_for_mean_normalized(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_hr_mean_feature.npy')

    @staticmethod
    def write_mean_normalized(subject_id, feature):
        mean_feature_path = HeartRateFeatureService.get_path_for_mean_normalized(subject_id)
        np.save(mean_feature_path, feature)

    @staticmethod
    def build(subject_id, valid_epochs):
        heart_rate_collection = HeartRateService.load_cropped(subject_id)
        return HeartRateFeatureService.build_from_collection(heart_rate_collection, valid_epochs)

    @staticmethod
    def build_mean(subject_id, valid_epochs):
        heart_rate_collection = HeartRateService.load_cropped(subject_id)
        return HeartRateFeatureService.build_mean_from_collection(heart_rate_collection, valid_epochs)

    @staticmethod
    def build_from_collection(heart_rate_collection, valid_epochs):
        heart_rate_features = []

        interpolated_timestamps, interpolated_hr = HeartRateFeatureService.interpolate_and_normalize(
            heart_rate_collection)

        min_timestamp = np.amin(interpolated_timestamps)

        for epoch in valid_epochs:
            if epoch.timestamp - min_timestamp < HeartRateFeatureService.WINDOW_SIZE:
                continue

            indices_in_range = HeartRateFeatureService.get_window(interpolated_timestamps, epoch)
            heart_rate_values_in_range = interpolated_hr[indices_in_range]

            feature = HeartRateFeatureService.get_feature(heart_rate_values_in_range)

            heart_rate_features.append(feature)

        return np.array(heart_rate_features)

    @staticmethod
    def build_mean_from_collection(heart_rate_collection, valid_epochs):
        raw_mean_features = []
        normalized_mean_features = []

        raw_timestamps, raw_hr = HeartRateFeatureService.interpolate_raw(heart_rate_collection)

        scalar = np.percentile(np.abs(raw_hr), 90)
        if scalar == 0:
            scalar = 1.0

        min_timestamp = np.amin(raw_timestamps)

        for epoch in valid_epochs:
            if epoch.timestamp - min_timestamp < HeartRateFeatureService.WINDOW_SIZE:
                continue

            indices_in_range = HeartRateFeatureService.get_window(raw_timestamps, epoch)
            hr_values_in_range = raw_hr[indices_in_range]

            raw_mean = np.mean(hr_values_in_range)
            normalized_mean = raw_mean / scalar

            raw_mean_features.append(raw_mean)
            normalized_mean_features.append(normalized_mean)

        return np.array(raw_mean_features), np.array(normalized_mean_features)

    @staticmethod
    def get_window(timestamps, epoch):
        start_time = epoch.timestamp - HeartRateFeatureService.WINDOW_SIZE
        end_time = epoch.timestamp + Epoch.DURATION
        timestamps_ravel = timestamps.ravel()
        indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                            timestamps.shape)
        return indices_in_range[0][0]

    @staticmethod
    def get_feature(heart_rate_values):
        # Use causal smoothing for consistency if needed, but here it's just std dev
        # If we wanted to weight recent values more, we'd need a weighted std dev
        return np.std(heart_rate_values)
        # return [np.std(heart_rate_values), np.mean(heart_rate_values)]

    @staticmethod
    def interpolate_and_normalize(heart_rate_collection):
        timestamps = heart_rate_collection.timestamps.flatten()
        heart_rate_values = heart_rate_collection.values.flatten()
        interpolated_timestamps = np.arange(np.amin(timestamps),
                                            np.amax(timestamps), 1)
        interpolated_hr = np.interp(interpolated_timestamps, timestamps, heart_rate_values)

        interpolated_hr = utils.convolve_with_dog(interpolated_hr, HeartRateFeatureService.WINDOW_SIZE)

        scalar = np.percentile(np.abs(interpolated_hr), 90)
        interpolated_hr = interpolated_hr / scalar

        return interpolated_timestamps, interpolated_hr

    @staticmethod
    def interpolate_raw(heart_rate_collection):
        timestamps = heart_rate_collection.timestamps.flatten()
        heart_rate_values = heart_rate_collection.values.flatten()
        interpolated_timestamps = np.arange(np.amin(timestamps),
                                            np.amax(timestamps), 1)
        interpolated_hr = np.interp(interpolated_timestamps, timestamps, heart_rate_values)

        return interpolated_timestamps, interpolated_hr
