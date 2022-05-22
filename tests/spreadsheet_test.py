import pytest
# importing the required libraries
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials



def test_spreadsheet():
    # define the scope
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    # add credentials to the account
    creds = ServiceAccountCredentials.from_json_keyfile_name('piglegsurgery-1987db83b363.json', scope)


    # authorize the clientsheet
    client = gspread.authorize(creds)

    # get the instance of the Spreadsheet
    sheet = client.open("https://docs.google.com/spreadsheets/d/1G55kXxcJ0PVaCApsDUJPGRkgoGEpiLaKn0sddRU-jzU/edit#gid=0")

    # get the first sheet of the Spreadsheet
    sheet_instance = sheet.get_worksheet(0)