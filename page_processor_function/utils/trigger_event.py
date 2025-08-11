import boto3
import os
import json

eventbridge = boto3.client("events")

BUS = os.getenv("BUS_NAME")


def put_in_bridge(message):

    response = eventbridge.put_events(
        Entries=[
            {
                "Source": "lambda.sender",
                "DetailType": "sending completion email",
                "Detail": json.dumps(message),
                "EventBusName": BUS,
            }
        ]
    )
    print("Event sent:", response)
    if response.get("FailedEntryCount", 0) > 0:
        return True
    else:
        return False
