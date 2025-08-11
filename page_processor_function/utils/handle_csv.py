import uuid
from django.http import HttpResponse


def convert_to_csv(df, extracted_data_dir):
    """
    Convert JSON data to CSV
    """
    csv_name = f"{uuid.uuid4()}.csv"
    csv_path = f"{extracted_data_dir}/{csv_name}"
    df.to_csv(csv_path, index=False)
    return csv_path, csv_name


def download_csv(csv_path):
    with open(csv_path, "rb") as f:
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = f"attachment; filename={csv_path}"
        response.write(f.read())

    return response
