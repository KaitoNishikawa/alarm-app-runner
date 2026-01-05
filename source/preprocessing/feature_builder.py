from source.constants import Constants
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService
from source.preprocessing.psg.psg_label_service import PSGLabelService
from source.preprocessing.psg.psg_service import PSGService
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService


class FeatureBuilder(object):

    @staticmethod
    def build(subject_id, data_path):
        if Constants.VERBOSE:
            print("Getting valid epochs...")
        valid_epochs = RawDataProcessor.get_valid_epochs(subject_id)

        psg_collection = PSGService.load_cropped(subject_id)
        activity_count_collection = ActivityCountService.load_cropped(subject_id)
        heart_rate_collection = HeartRateService.load_cropped(subject_id)

        start_time = max(psg_collection.data[0].epoch.timestamp,
                         activity_count_collection.timestamps[0],
                         heart_rate_collection.timestamps[0])

        if Constants.VERBOSE:
            print(f"Global Start Time: {start_time}")
        
        valid_epochs = [e for e in valid_epochs if e.timestamp - start_time >= ActivityCountFeatureService.WINDOW_SIZE]

        original_start_time = PSGService.get_original_start_time(subject_id, data_path)
        if Constants.VERBOSE:
            print(f"Original Start Time: {original_start_time}")

        if Constants.VERBOSE:
            print("Building features...")
        FeatureBuilder.build_labels(subject_id, valid_epochs)
        FeatureBuilder.build_from_wearables(subject_id, valid_epochs)
        FeatureBuilder.build_from_time(subject_id, valid_epochs, original_start_time)

    @staticmethod
    def build_labels(subject_id, valid_epochs):
        psg_labels = PSGLabelService.build(subject_id, valid_epochs)
        PSGLabelService.write(subject_id, psg_labels)

    @staticmethod
    def build_from_wearables(subject_id, valid_epochs):

        count_feature = ActivityCountFeatureService.build(subject_id, valid_epochs)
        heart_rate_feature = HeartRateFeatureService.build(subject_id, valid_epochs)
        hr_mean_raw_feature, hr_mean_normalized_feature = HeartRateFeatureService.build_mean(subject_id, valid_epochs)
        ActivityCountFeatureService.write(subject_id, count_feature)
        HeartRateFeatureService.write(subject_id, heart_rate_feature)
        # HeartRateFeatureService.write_mean_raw(subject_id, hr_mean_raw_feature)
        HeartRateFeatureService.write_mean_normalized(subject_id, hr_mean_normalized_feature)

    @staticmethod
    def build_from_time(subject_id, valid_epochs, start_time=None):

        if Constants.INCLUDE_CIRCADIAN:
            circadian_feature = TimeBasedFeatureService.build_circadian_model(subject_id, valid_epochs)
            TimeBasedFeatureService.write_circadian_model(subject_id, circadian_feature)

        cosine_feature = TimeBasedFeatureService.build_cosine(valid_epochs, start_time)
        time_feature = TimeBasedFeatureService.build_time(valid_epochs, start_time)

        TimeBasedFeatureService.write_cosine(subject_id, cosine_feature)
        TimeBasedFeatureService.write_time(subject_id, time_feature)
