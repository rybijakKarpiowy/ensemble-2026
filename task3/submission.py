import os
import csv
import random

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = "task3"

API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")

CSV_FILE="submission.csv"


def main():
    if not API_TOKEN:
        raise ValueError(
            "TEAM_TOKEN not provided. Define TEAM_TOKEN in .env"
        )

    if not SERVER_URL:
        raise ValueError(
            "SERVER_URL not defined. Define SERVER_URL in .env"
        )

    headers = {
        "X-API-Token": API_TOKEN
    }

    # Important, the name of key in files - "csv_file" must be exact
    response = requests.post(
        f"{SERVER_URL}/{ENDPOINT}",
        files={"csv_file": open(CSV_FILE, "rb")},
        headers=headers
    )

    try:
        data = response.json()
    except Exception:
        data = response.text

    print("response:", response.status_code, data)

if __name__ == "__main__":
    main()