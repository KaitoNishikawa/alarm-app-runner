import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants
from source.preprocessing.motion.motion_collection import MotionCollection


class MotionService(object):

    @staticmethod
    def load_raw(subject_id, data_path):
        raw_motion_path = MotionService.get_raw_file_path(subject_id, data_path)
        motion_array = np.load(str(raw_motion_path))
        motion_array = utils.remove_repeats(motion_array)
        return MotionCollection(subject_id=subject_id, data=motion_array)

    @staticmethod
    def load_cropped(subject_id):
        cropped_motion_path = MotionService.get_cropped_file_path(subject_id)
        motion_array = np.load(str(cropped_motion_path))
        return MotionCollection(subject_id=subject_id, data=motion_array)

    @staticmethod
    def load(motion_file, delimiter=' '):
        motion_array = pd.read_csv(str(motion_file), delimiter=delimiter).values
        return motion_array

    @staticmethod
    def write(motion_collection):
        motion_output_path = MotionService.get_cropped_file_path(motion_collection.subject_id)
        np.save(motion_output_path, motion_collection.data)

    @staticmethod
    def crop(motion_collection, interval):
        subject_id = motion_collection.subject_id
        timestamps = motion_collection.timestamps
        valid_indices = ((timestamps >= interval.start_time)
                         & (timestamps < interval.end_time)).nonzero()[0]

        cropped_data = motion_collection.data[valid_indices, :]
        return MotionCollection(subject_id=subject_id, data=cropped_data)

    @staticmethod
    def get_cropped_file_path(subject_id):
        return Constants.CROPPED_FILE_PATH.joinpath(subject_id + "_cleaned_motion.npy")

    @staticmethod
    def get_raw_file_path(subject_id, data_path):
        project_root = utils.get_project_root()
        return project_root.joinpath(data_path + '/acceleration/' + subject_id + '_acceleration.npy')
