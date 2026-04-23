from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CONCENTRADO_PATH = REPO_ROOT / "data" / "onedrive" / "concentrado_base_CMAT.xlsx"

def read_concentrado():
    years = [19, 20, 21, 22, 23, 24]

    # create a dictionany to hold dataframes
    df = {}
    for year in years:
        df[year] = pd.read_excel(CONCENTRADO_PATH, sheet_name=f'20{year}')
        df[year]['Year'] = 2000 + year

    frames = [df[y] for y in years if y in df]
    out = pd.concat(frames, ignore_index=True)
    del frames, df  # if you’re done with them

    return out
