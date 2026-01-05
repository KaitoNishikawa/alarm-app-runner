import os
import sys
import json
import boto3
import joblib
import requests
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request

current_dir = os.path.dirname(os.path.abspath(__file__))
docker_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, docker_root)

from endpoint_stuff.handle_data import HandleData
from source.preprocessing.preprocessing_runner import PreprocessingRunner

load_dotenv()
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

app = Flask(__name__)

s3 = boto3.client(
    "s3",
    aws_access_key_id = access_key,
    aws_secret_access_key = secret_access_key,
)
bucket_name = 's3-smart-alarm-app'

model_path = "saved_model/Random_Forest.joblib"
clf = joblib.load(model_path)

@app.route('/hello')
def hello_world():
    return jsonify(message='Hello World')

@app.route('/s3-webhook', methods=['POST'], strict_slashes=False)
def s3_webhook():
    # SNS sends JSON data in the request body
    try:
        data = json.loads(request.data)
    except Exception as e:
        print(f"Error parsing request data: {e}")
        return "Invalid JSON", 400
    
    # SNS sends a header to identify the message type
    sns_message_type = request.headers.get('x-amz-sns-message-type')
    
    if sns_message_type == 'SubscriptionConfirmation':
        # AWS SNS requires us to visit the SubscribeURL to confirm the subscription
        subscribe_url = data.get('SubscribeURL')
        if subscribe_url:
            requests.get(subscribe_url)
            print(f"Confirmed subscription at {subscribe_url}")
            return "Subscription confirmed", 200
            
    elif sns_message_type == 'Notification':
        # This is the actual S3 event notification
        # The S3 event data is stringified inside the 'Message' field
        try:
            message = json.loads(data.get('Message'))
        except Exception as e:
            print(f"Error parsing SNS message: {e}")
            return "Invalid Message format", 400
        
        # S3 events are inside the 'Records' list
        if 'Records' in message:
            sessions_to_process = set()

            for record in message['Records']:
                bucket_name = record['s3']['bucket']['name']
                object_key = record['s3']['object']['key']
                event_name = record['eventName']

                # Ignore prediction files and non-user data to avoid infinite loops
                if 'predictions' in object_key:
                    print(f"Ignoring S3 Event: {event_name} for object {object_key}")
                    continue

                print(f"New S3 Event: {event_name} in bucket {bucket_name} for object {object_key}")
                
                # Replace 'users' with 'user_data' in the path
                modified_key = object_key.replace('users/', 'user_data/', 1)
                local_path = os.path.join('data', modified_key)
                
                # Create the directory structure (excluding the filename)
                local_dir = os.path.dirname(local_path)
                os.makedirs(local_dir, exist_ok=True)
                
                s3.download_file(bucket_name, object_key, local_path)
                print(f"Downloaded {object_key} to {local_path}")
                
                dir_name = os.path.dirname(local_path)
                HandleData.concat_npy_files(dir_name)
                
                # Add the parent session directory to the set of sessions to check
                sessions_to_process.add(os.path.dirname(dir_name))

            for session_dir in sessions_to_process:
                if HandleData.is_session_ready(session_dir):
                    print(f"Session {session_dir} is ready. Running preprocessing...")
                    PreprocessingRunner.run_preprocessing('0721', session_dir)
                    feature_df = HandleData.load_files_into_df(session_dir)
                    feature_df.to_csv('test.csv')
                    predictions = HandleData.make_predictions(feature_df, clf, session_dir)
                    HandleData.upload_predictions_to_s3(predictions, bucket_name, object_key, s3)
        
        return "Notification received", 200

    return "OK", 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
