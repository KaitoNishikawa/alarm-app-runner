import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants
from source.preprocessing.heart_rate.heart_rate_collection import HeartRateCollection


class HeartRateService(object):

    @staticmethod
    def load_raw(subject_id, data_path):
        raw_hr_path = HeartRateService.get_raw_file_path(subject_id, data_path)
        heart_rate_array = np.load(str(raw_hr_path))
        heart_rate_array = utils.remove_repeats(heart_rate_array)
        return HeartRateCollection(subject_id=subject_id, data=heart_rate_array)

    @staticmethod
    def load_cropped(subject_id):
        cropped_hr_path = HeartRateService.get_cropped_file_path(subject_id)
        heart_rate_array = np.load(str(cropped_hr_path))
        return HeartRateCollection(subject_id=subject_id, data=heart_rate_array)

    @staticmethod
    def load(hr_file, delimiter=" "):
        heart_rate_array = pd.read_csv(str(hr_file), delimiter=delimiter).values
        return heart_rate_array

    @staticmethod
    def write(heart_rate_collection):
        hr_output_path = HeartRateService.get_cropped_file_path(heart_rate_collection.subject_id)
        np.save(hr_output_path, heart_rate_collection.data)

    @staticmethod
    def crop(heart_rate_collection, interval):
        subject_id = heart_rate_collection.subject_id
        timestamps = heart_rate_collection.timestamps
        valid_indices = ((timestamps >= interval.start_time)
                         & (timestamps < interval.end_time)).nonzero()[0]

        cropped_data = heart_rate_collection.data[valid_indices, :]
        return HeartRateCollection(subject_id=subject_id, data=cropped_data)

    @staticmethod
    def get_cropped_file_path(subject_id):
        return Constants.CROPPED_FILE_PATH.joinpath(subject_id + "_cleaned_hr.npy")

    @staticmethod
    def get_raw_file_path(subject_id, data_path):
        # heart_rate_dir = utils.get_project_root().joinpath('data/heart_rate/')
        project_root = utils.get_project_root()
        return project_root.joinpath(data_path + '/heartrate/' + subject_id + '_heartrate.npy')
