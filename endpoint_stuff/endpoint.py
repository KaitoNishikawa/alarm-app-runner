import os
import json
import boto3
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request

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
            for record in message['Records']:
                bucket_name = record['s3']['bucket']['name']
                object_key = record['s3']['object']['key']
                event_name = record['eventName']
                print(f"New S3 Event: {event_name} in bucket {bucket_name} for object {object_key}")
                
                # Replace 'users' with 'user_data' in the path
                modified_key = object_key.replace('users/', 'user_data/', 1)
                local_path = os.path.join('data', modified_key)
                
                # Create the directory structure (excluding the filename)
                local_dir = os.path.dirname(local_path)
                os.makedirs(local_dir, exist_ok=True)
                
                s3.download_file(bucket_name, object_key, local_path)
                print(f"Downloaded {object_key} to {local_path}")
        
        return "Notification received", 200

    return "OK", 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
