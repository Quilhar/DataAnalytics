def get_lowercase_columns(columns):
    
    headers = {}
    
    columns = list(columns)
    
    for col in columns:
        
        headers[col] = col.lower()
        
    return headers

def find_nulls(df, col_name):
    null_dict = {}
    
    null_index = df.loc[df[col_name].isnull()].index
    
    null_dict[col_name] = null_index
    
    return null_dict