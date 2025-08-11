import boto3, os, json, logging

log = logging.getLogger()
log.setLevel(logging.INFO)

eventbridge = boto3.client("events")
BUS = os.getenv("BUS_NAME")


def put_in_bridge(message: dict) -> bool:
    if not BUS:
        log.error("Missing BUS_NAME env var")
        return False

    log.info(f"Event bus: {BUS}")
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
        log.info(f"PutEvents response: {resp}")
        return resp.get("FailedEntryCount", 0) == 0
    except Exception as e:
        log.exception("PutEvents failed")
        return False
