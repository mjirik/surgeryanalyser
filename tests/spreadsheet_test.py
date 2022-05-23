import pytest
# importing the required libraries
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
from typing import Union


def google_spreadsheet_append(title: str, creds, data:Union[pd.DataFrame, dict], scope=None):
    # define the scope

    # https://www.analyticsvidhya.com/blog/2020/07/read-and-update-google-spreadsheets-with-python/

    if type(data) in (dict):
        df_novy = pd.DataFrame(data)
    else:
        df_novy = data
    if scope is None:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    # add credentials to the account
    if type(creds) in (str, Path):
        creds = ServiceAccountCredentials.from_json_keyfile_name(Path(creds), scope)

    # authorize the clientsheet
    client = gspread.authorize(creds)

    # get the instance of the Spreadsheet
    sheet = client.open(title)

    # get the first sheet of the Spreadsheet
    sheet_instance = sheet.get_worksheet(0)

    # get all the records of the data
    records_data = sheet_instance.get_all_records()

    # convert the json to dataframe
    records_df = pd.DataFrame.from_dict(records_data)

    # view the top records
    records_df.head()

    # # number of runs by each batsman
    # runs = records_df.groupby(['Prvni sloupec'])['Druhy'].count().reset_index()
    # runs
    #
    # # print(records_data.values.to_list())
    # records_df.keys()
    #
    # sheet_instance.append_rows([['dd', 5]])
    #
    #
    # records_df.keys()
    # novy.keys()
    #
    # new_keys = []
    # for key in novy.keys():
    #     if key not in records_df.keys():
    #         new_keys.append(key)
    #
    # sheet_instance.add_cols(len(new_keys))
    # for key in new_keys:
    #     sheet_instance.insert_cols(values=[[key]], col=(len(records_df.keys()) + 1))

    records_data = sheet_instance.get_all_records()

    # convert the json to dataframe
    records_df = pd.DataFrame.from_dict(records_data)
    df_empty = pd.DataFrame(columns=records_df.keys())

    df_out = pd.concat([df_empty, df_novy], axis=0)

    # remove NaN
    df_out2 = df_out.where(pd.notnull(df_out), None)

    sheet_instance.append_rows(df_out2.values.tolist())
    print("asd")



def test_spreadsheet():
    creds_file = Path('piglegsurgery-1987db83b363.json')
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)

    novy = {"Prvni sloupec": ["aa"], "Treti": [55]}
    df_novy = pd.DataFrame(novy)

    google_spreadsheet_append(
        title="Pigleg Surgery Stats",
        creds=creds,
        dataframe=df_novy
    )



