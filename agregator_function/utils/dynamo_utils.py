import boto3
import os
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError

# === Dynamo DB ===
dynamodb = boto3.resource("dynamodb")
d_table = os.getenv("JOB_STATUS_TABLE")
table = dynamodb.Table(d_table)  # Name passed via env var


def get_items_from_record_0_table(job_id, page_number):
    status = ["PROCESSING_PAGES"]
    next = "ProcessPages"

    try:
        response = table.query(
            KeyConditionExpression=Key("job_id").eq(job_id)
            & Key("page_number").eq(page_number),
            FilterExpression=Attr("status").is_in(status) & Attr("next").eq(next),
        )
        items = response.get("Items", [])

        return items

    except ClientError as e:
        print(f"DynamoDB read error: {e}")
        raise e


def get_items_from_record_table(job_id, page_number):
    status = ["FAILED", "COMPLETED"]
    next = "final_aggregator"

    try:
        response = table.query(
            KeyConditionExpression=Key("job_id").eq(job_id)
            & Key("page_number").eq(page_number),
            FilterExpression=Attr("status").is_in(status) & Attr("next").eq(next),
        )
        items = response.get("Items", [])

        return items

    except ClientError as e:
        print(f"DynamoDB read error: {e}")
        raise e


def update_record_table(job_id: int, page_number: int, update_data: dict):
    """
    Update an existing record in DynamoDB without overwriting the whole item.
    Skips None values to avoid deleting attributes unintentionally.
    """
    if not job_id:
        raise ValueError("job_id is required to update the record")

    if not page_number:
        raise ValueError("page_number is required to update the record")

    if not update_data:
        raise ValueError("update_data must contain at least one field to update")

    # Remove keys that are None (so we don't accidentally nullify them)
    filtered_data = {k: v for k, v in update_data.items() if v is not None}

    if not filtered_data:
        raise ValueError("No valid fields to update after filtering None values")

    # Build UpdateExpression
    update_expr_parts = []
    expr_attr_values = {}
    expr_attr_names = {}

    for key, value in filtered_data.items():
        expr_attr_names[f"#{key}"] = key
        expr_attr_values[f":{key}"] = value
        update_expr_parts.append(f"#{key} = :{key}")

    update_expression = "SET " + ", ".join(update_expr_parts)

    try:
        response = table.update_item(
            Key={"job_id": job_id, "page_number": page_number},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expr_attr_names,
            ExpressionAttributeValues=expr_attr_values,
        )
        return response
    except ClientError as e:
        raise Exception(f"Failed to update record {job_id}: {e}")
