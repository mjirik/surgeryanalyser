from pathlib import Path
from typing import Union

# importing the required libraries
import gspread
import pandas as pd
import pytest
from oauth2client.service_account import ServiceAccountCredentials

from piglegsurgeryweb.uploader.data_tools import google_spreadsheet_append


def test_spreadsheet():
    creds_file = Path("piglegsurgery-1987db83b363.json")
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)

    novy = {"Prvni sloupec": ["aa"], "Treti": [55]}
    novy = {"loupec": ["aa"]}
    df_novy = pd.DataFrame(novy)

    google_spreadsheet_append(title="Pigleg Surgery Stats", creds=creds, data=df_novy)
