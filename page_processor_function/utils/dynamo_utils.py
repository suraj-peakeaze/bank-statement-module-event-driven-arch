import boto3
import os
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError

# Dynamo DB
dynamodb = boto3.resource("dynamodb")
d_table = os.getenv("JOB_STATUS_TABLE")
table = dynamodb.Table(d_table)


def get_items_from_record_table(job_id, page_number):
    """
    Fetch items for a given job_id and page_number that are either FAILED or PROCESSING_PAGES,
    and have next='ProcessPages'.
    """
    status = ["FAILED", "PROCESSING_PAGES"]
    next_stage = "ProcessPages"

    try:
        response = table.query(
            KeyConditionExpression=Key("job_id").eq(int(job_id))
            & Key("page_number").eq(int(page_number)),
            FilterExpression=Attr("status").is_in(status) & Attr("next").eq(next_stage),
        )
        return response.get("Items", [])
    except ClientError as e:
        print(f"DynamoDB read error: {e}")
        raise


def update_record_table(job_id, page_number, update_data: dict):
    """
    Update an existing record in DynamoDB without overwriting the whole item.
    Skips None values and primary keys to avoid deleting attributes unintentionally.
    """
    if job_id is None:
        raise ValueError("job_id is required")
    if page_number is None:
        raise ValueError("page_number is required")
    if not update_data:
        raise ValueError("update_data must contain at least one field to update")

    filtered_data = {
        k: v
        for k, v in update_data.items()
        if v is not None and k not in ("job_id", "page_number")
    }
    if not filtered_data:
        raise ValueError("No valid fields to update after filtering")

    update_expr_parts = []
    expr_attr_values = {}
    expr_attr_names = {}

    for key, value in filtered_data.items():
        expr_attr_names[f"#{key}"] = key
        expr_attr_values[f":{key}"] = value
        update_expr_parts.append(f"#{key} = :{key}")

    update_expression = "SET " + ", ".join(update_expr_parts)

    try:
        return table.update_item(
            Key={"job_id": int(job_id), "page_number": page_number},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expr_attr_names,
            ExpressionAttributeValues=expr_attr_values,
        )
    except ClientError as e:
        raise Exception(f"Failed to update record {job_id}, page {page_number}: {e}")
