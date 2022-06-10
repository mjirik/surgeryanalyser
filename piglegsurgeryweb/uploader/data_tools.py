import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
from typing import Optional, Union
from loguru import logger



def google_spreadsheet_append(title: str, creds, data:Union[pd.DataFrame, dict], scope=None, sheet_index=0) -> pd.DataFrame:
    # define the scope

    # https://www.analyticsvidhya.com/blog/2020/07/read-and-update-google-spreadsheets-with-python/

    if type(data) == dict:
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
    sheet_instance = sheet.get_worksheet(sheet_index)

    # get all the records of the data
    # records_data = sheet_instance.get_all_records()

    # convert the json to dataframe
    # records_df = pd.DataFrame.from_dict(records_data)

    # view the top records
    # records_df.head()
    records_data = sheet_instance.get_all_records()

    # convert the json to dataframe
    records_df = pd.DataFrame.from_dict(records_data)
    df_concat = pd.concat([records_df, df_novy], axis=0, ignore_index=True)
    df_empty = pd.DataFrame(columns=df_concat.keys())

    df_out = pd.concat([df_empty, df_novy], axis=0)

    # remove NaN
    df_out2 = df_out.where(pd.notnull(df_out), None)
    logger.debug(f"appended row={df_out2.values.tolist()}")
    sheet_instance.append_rows(df_out2.values.tolist())

    # update  header
    for i, key in enumerate(df_out2.keys()):
        val = sheet_instance.cell(1, i + 1).value
        if val != key:
            sheet_instance.update_cell(1, i + 1, key)

    return df_out2

