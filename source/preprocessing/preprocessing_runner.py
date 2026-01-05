import time
import os
from pathlib import Path

from source.analysis.figures.data_plot_builder import DataPlotBuilder
from source.analysis.setup.subject_builder import SubjectBuilder
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.feature_builder import FeatureBuilder
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.time.circadian_service import CircadianService

class PreprocessingRunner:
    @staticmethod
    def run_preprocessing(subject, data_path):
        start_time = time.time()
        
        cropped_path = os.path.join(data_path, 'outputs/cropped/')
        features_path = os.path.join(data_path, 'outputs/features/')

        os.makedirs(cropped_path, exist_ok=True)
        os.makedirs(features_path, exist_ok=True)

        Constants.update('CROPPED_FILE_PATH', Path(cropped_path))
        Constants.update('FEATURE_FILE_PATH', Path(features_path))

        print("Cropping data from subject " + str(subject) + "...")
        RawDataProcessor.crop_all(subject, data_path)

        if Constants.INCLUDE_CIRCADIAN:
            ActivityCountService.build_activity_counts()  # This uses MATLAB, but has been replaced with a python implementation
            CircadianService.build_circadian_model()      # Both of the circadian lines require MATLAB to run
            CircadianService.build_circadian_mesa()       # INCLUDE_CIRCADIAN = False by default because most people don't have MATLAB

        
        FeatureBuilder.build(subject, data_path)

        end_time = time.time()
        print("Execution took " + str(end_time - start_time) + " seconds")


# subject_ids = SubjectBuilder.get_all_subject_ids()
# PreprocessingRunner.run_preprocessing('893', 'data/user_data/0001/20260102_184054')

# for subject_id in subject_ids:
#     DataPlotBuilder.make_data_demo(subject_id, False)
