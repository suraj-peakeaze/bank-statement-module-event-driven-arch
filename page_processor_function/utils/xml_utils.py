import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os


def file_to_xml(file_path):
    """
    Converts a CSV or Excel file to an XML string based on the specified structure.

    Args:
        file_path (str): The path to the file (CSV or Excel).

    Returns:
        str: A pretty-printed XML string, or None if an error occurs.
    """
    try:
        # Read the file into a pandas DataFrame
        if file_path.endswith(".csv"):
            # Try utf-8 first, then latin-1 if utf-8 fails for CSV
            try:
                df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                print("UTF-8 decoding failed for CSV, trying latin-1...")
                df = pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip")
        elif file_path.endswith(".xlsx"):
            print("Reading Excel file...")
            df = pd.read_excel(file_path)
        else:
            print("Error: Unsupported file format. Please upload a .csv or .xlsx file.")
            return None

        # Create the root element for the XML
        table_element = ET.Element("table")
        rows_element = ET.SubElement(table_element, "rows")

        # Iterate through each row in the DataFrame
        for r_idx, row_data in df.iterrows():
            # Create a row element like <row_0>, <row_1>, etc.
            row_index_element = ET.SubElement(rows_element, f"row_{r_idx}")

            # Iterate through each column in the row
            for c_idx, value in enumerate(row_data):
                # Create a column element like <col_0>, <col_1>, etc.
                col_index_element = ET.SubElement(row_index_element, f"col_{c_idx}")
                col_index_element.text = str(value)  # Ensure value is a string

        # Create an ElementTree object
        tree = ET.ElementTree(table_element)

        # Pretty-print the XML for better readability
        # ET.tostring provides a byte string, decode it to utf-8
        rough_string = ET.tostring(table_element, "utf-8", method="xml")
        reparsed_xml = minidom.parseString(rough_string)
        pretty_xml_str = reparsed_xml.toprettyxml(indent="  ")

        return pretty_xml_str

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The uploaded file is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def run_file_to_xml_converter(file_path, xml_dir):
    """
    Main function to handle file upload, conversion, and download in Colab.
    """
    if not file_path:
        print("No file was uploaded. Exiting.")
        return

    # Get the name of the uploaded file
    file_name = os.path.basename(file_path)
    print(f"File '{file_name}' received successfully.")

    # Perform the conversion
    xml_output = file_to_xml(file_path)

    if xml_output:
        # Define the output XML file name
        output_xml_name = os.path.join(xml_dir, os.path.splitext(file_name)[0] + ".xml")

        # Save the XML string to a file
        with open(output_xml_name, "w", encoding="utf-8") as f:
            f.write(xml_output)

        print(f"\nXML file '{output_xml_name}' created successfully.")
        return xml_output
    else:
        print("XML conversion failed. Please check the console for errors.")
