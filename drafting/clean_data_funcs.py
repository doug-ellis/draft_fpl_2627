import unidecode
import pandas as pd

def clean_name(name):
    if pd.isna(name):
        return ""
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    name = unidecode.unidecode(name)

    return name

def combine_clean_names(df, first_name_col, second_name_col):
    df['full_name'] = df[first_name_col] + '_' + df[second_name_col]
    df['full_name'] = df['full_name'].apply(clean_name)
    
    return df

