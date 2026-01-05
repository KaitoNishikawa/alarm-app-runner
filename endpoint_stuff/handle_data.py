import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import os

class HandleData:
    @staticmethod
    def concat_npy_files(dir_path):
        file_name = ''
        if 'heartrate' in dir_path:
            file_name = '0721_heartrate.npy'
        elif 'acceleration' in dir_path:
            file_name = '0721_acceleration.npy'
        else:
            return

        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.npy') and f != file_name])
        if not files:
            return 

        data_list = [np.load(os.path.join(dir_path, f)) for f in files]
        concatenated_data = np.concatenate(data_list, axis=0)
        
        np.save(os.path.join(dir_path, file_name), concatenated_data)

        if 'acceleration' in dir_path:
            timestamp = concatenated_data[-1][0]
            HandleData.create_label_file(dir_path, timestamp)

    @staticmethod
    def is_session_ready(session_dir):
        accel_path = os.path.join(session_dir, 'acceleration', '0721_acceleration.npy')
        hr_path = os.path.join(session_dir, 'heartrate', '0721_heartrate.npy')
        
        if not (os.path.exists(accel_path) and os.path.exists(hr_path)):
            return False
            
        try:
            # Use mmap_mode='r' to read only metadata/necessary parts
            accel_data = np.load(accel_path, mmap_mode='r')
            hr_data = np.load(hr_path, mmap_mode='r')
            
            if accel_data.size == 0 or hr_data.size == 0:
                return False

            # Assuming shape (N, 4) or (N, 2) where col 0 is timestamp
            accel_last_ts = accel_data[-1, 0]
            hr_last_ts = hr_data[-1, 0]

            if hr_last_ts < 300 or accel_last_ts < 300:
                return False
            
            # Allow for some small drift/jitter (e.g. < 5 seconds)
            return abs(accel_last_ts - hr_last_ts) < 40.0
            
        except Exception as e:
            print(f"Error checking session readiness: {e}")
            return False

    @staticmethod
    def create_label_file(dir_path, timestamp):
        labels_dir = os.path.join(os.path.dirname(dir_path), 'labels')
        os.makedirs(labels_dir, exist_ok=True)
        file_path = os.path.join(labels_dir, '0721_labeled_sleep.npy')
        
        num_labels = round(timestamp / 30) + 1
        timestamps = np.arange(0, num_labels * 30, 30).astype('int')
        labels = np.zeros(len(timestamps)).astype('int')
        label_array = np.column_stack((timestamps, labels))

        np.save(file_path, label_array)
        # np.savetxt('test.out', label_array)
        print(f"Created label file at {file_path}")
    
    @staticmethod
    def load_files_into_df(dir_path):
        dir_path = os.path.join(dir_path, 'outputs', 'features')

        cosine_feature = np.load(os.path.join(dir_path, '0721_cosine_feature.npy'))
        count_feature = np.load(os.path.join(dir_path, '0721_count_feature.npy'))
        hr_std_feature = np.load(os.path.join(dir_path, '0721_hr_feature.npy'))
        hr_mean_feature = np.load(os.path.join(dir_path, '0721_hr_mean_feature.npy'))
        time_feature = np.load(os.path.join(dir_path, '0721_time_feature.npy'))

        df = pd.DataFrame({
            'cosine_feature': cosine_feature,
            'count_feature': count_feature,
            'hr_std': hr_std_feature,
            'hr_mean': hr_mean_feature,
            'time_feature': time_feature,
        })

        df['count_feature_lag_1'] = df['count_feature'].shift(1)
        df['count_feature_lag_2'] = df['count_feature'].shift(2)

        df['hr_std_lag_1'] = df['hr_std'].shift(1)
        df['hr_std_lag_2'] = df['hr_std'].shift(2)

        df['hr_mean_lag_1'] = df['hr_mean'].shift(1)
        df['hr_mean_lag_2'] = df['hr_mean'].shift(2)

        df['hr_mean_delta'] = df['hr_mean'] - df['hr_mean'].shift(2)
        df = df.iloc[2:].reset_index(drop=True)

        if not df.empty:
            scaler = StandardScaler()
            df['hr_mean_delta'] = scaler.fit_transform(df[['hr_mean_delta']])

        return df 
    
    @staticmethod
    def make_predictions(feature_df, model, session_dir):
        if feature_df.empty:
            return np.array([])

        predictions = model.predict(feature_df)
        np.savetxt('predictions.out', predictions, fmt='%d')

        save_path = os.path.join(session_dir, 'outputs', 'predictions')
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, '0721_predictions.npy'), predictions)

        return predictions

    @staticmethod
    def upload_predictions_to_s3(predictions, bucket_name, dir_path, s3):
        if predictions.size == 0:
            return  
        
        # Ensure we use forward slashes for S3 keys regardless of OS
        parts = dir_path.replace('\\', '/').split('/')
        if len(parts) >= 3:
            # Get the session directory (e.g., users/0001/20260102_142813)
            session_prefix = '/'.join(parts[:3])
            prediction_key = f"{session_prefix}/predictions/0721_predictions.json"
            
            try:
                s3.put_object(
                    Bucket=bucket_name,
                    Key=prediction_key,
                    Body=json.dumps(predictions.tolist()),
                    ContentType='application/json'
                )
                print(f"Uploaded predictions to s3://{bucket_name}/{prediction_key}")
            except Exception as e:
                print(f"Failed to upload predictions: {e}")


