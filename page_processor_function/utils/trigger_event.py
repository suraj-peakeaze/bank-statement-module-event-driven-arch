import boto3, os, json
from utils.logger_config import get_logger

logger = get_logger(__name__)

eventbridge = boto3.client("events")
BUS = os.getenv("BUS_NAME")


def put_in_bridge(message: dict) -> bool:
    if not BUS:
        logger.error("Missing BUS_NAME env var")
        return False

    logger.info(f"Event bus: {BUS}")
    try:
        resp = eventbridge.put_events(
            Entries=[
                {
                    "Source": "custom.processor",
                    "DetailType": "trigger-lamb2",
                    "Detail": json.dumps(message),
                    "EventBusName": BUS,  # name or ARN both work
                }
            ]
        )
        logger.info(f"PutEvents response: {resp}")
        return resp.get("FailedEntryCount", 0) == 0
    except Exception as e:
        logger.exception("PutEvents failed")
        return False
