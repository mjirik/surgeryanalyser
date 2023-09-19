import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
from typing import Optional, Union
from loguru import logger



def flatten_dict(dct:dict, parent_key:str='', sep:str='_') -> dict:
    """
    Flatten nested dictionary
    :param dct: nested dictionary
    :param parent_key: parent key
    :param sep: separator
    :return: flattened dictionary
    """
    items = []
    for k, v in dct.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
    return dict(items)


def remove_empty_lists(dct:dict) -> dict:
    """
    Remove empty lists from dictionary
    :param dct: dictionary
    :return: dictionary without empty lists
    """
    return {k: v for k, v in dct.items() if v != []}


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
    df_out2 = df_out2.fillna('')
    logger.debug(f"appended keys={list(df_out2.keys())}")
    logger.debug(f"appended row={df_out2.values.tolist()}")
    sheet_instance.append_rows(df_out2.values.tolist())

    # update  header
    cell_list = sheet_instance.range(1, 1, 1, len(df_out2.keys()))
    sheet_instance.update_cells(cell_list, )

    for i, key in enumerate(df_out2.keys()):
        val = cell_list[i].value
        # val = sheet_instance.cell(1, i + 1).value
        if val != key:
            cell_list[i].value = key
            # sheet_instance.update_cell(1, i + 1, key)

    sheet_instance.update_cells(cell_list)

    return df_out2


def remove_iterables_from_dict(dct:dict) -> dict:
    """
    Remove iterables from dictionary
    :param dct: dictionary
    :return: dictionary without iterables
    """
    return {k: v for k, v in dct.items() if not hasattr(v, '__iter__')}
