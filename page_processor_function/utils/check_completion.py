import os
import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.client import ClientError

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.getenv("JOB_STATUS_TABLE"))


def check_if_completed(job_id, page):
    status = ["COMPLETED"]
    next = "final_aggregator"

    try:
        response = table.query(
            KeyConditionExpression=Key("job_id").eq(job_id)
            & Key("page_number").eq(page),
            FilterExpression=Attr("status").is_in(status) & Attr("next").eq(next),
        )
        if response.get("Items"):
            return True
        else:
            return False

    except ClientError as e:
        raise (f"DynamoDB read error: {e}")


def check_if_email_sent(job_id):
    status = ["COMPLETED"]

    try:
        response = table.query(
            KeyConditionExpression=Key("job_id").eq(job_id) & Key("page_number").eq(0),
            FilterExpression=Attr("status").is_in(status)
            & Attr("is_email_sent").eq(True),
        )
        if response.get("Items"):
            return True
        else:
            return False

    except ClientError as e:
        raise (f"DynamoDB read error: {e}")
