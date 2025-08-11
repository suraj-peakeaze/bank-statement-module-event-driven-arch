# Automated Bank Statement Processing Pipeline

This project is designed to automatically process Bank Statements, extract information, and notify users when the process is complete. It's built using a series of serverless functions that work together.

## How It Works

The process happens in three main steps:

1.  **Splitting the Bank Statement:** When a PDF Bank Statement is added, the first function (`split_pdf_function`) splits it into single pages.

2.  **Processing Each Page:** Each page is then sent to the second function (`page_processor_function`). This function reads the content of the page using Optical Character Recognition (OCR) and extracts the important information.

3.  **Aggregating the Results:** Once all pages have been processed, the final function (`agregator_function`) collects all the extracted information, saves it, and sends an email to notify that the Bank Statement has been processed.

## Core Components

*   `split_pdf_function/`: Contains the code for splitting PDF Bank Statements.
*   `page_processor_function/`: Holds the logic for performing OCR and extracting data from each page.
*   `agregator_function/`: Responsible for combining the results from all pages and sending a final notification.

Each function has its own `requirements.txt` to manage its dependencies and a `dockerfile` to package it for deployment.
