def get_lowercase_columns(columns):
    
    headers = {}
    
    columns = list(columns)
    
    for col in columns:
        
        headers[col] = col.lower()
        
    return headers

def find_nulls(dataframe):
    
    nulls = {}
    
    for i in dataframe.columns:
        
        nulls[i] = dataframe.loc[dataframe[i].isnull()].index.tolist()
        
    return nulls

def find_duplicated_values(dataframe):
    
    duplicated_rows = {}
    for i in dataframe.columns:
        print(f"{i} has {dataframe[i].duplicated().sum()} duplicated values")

        if dataframe[i].duplicated().sum() > 0:
            duplicated_rows.update({i: dataframe.loc[dataframe.duplicated(subset = i)]})
    
    return duplicated_rows

def drop_duplicates(df):
    
    for i in df.columns:
        
        df.drop_duplicates(keep ='first', subset = i, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    
    return df