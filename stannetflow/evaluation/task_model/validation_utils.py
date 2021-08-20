from datetime import datetime

def ip_split(df, col_name, norm=False):
    # new data frame with split value columns
    new = df[col_name].str.split(".", expand = True)
    for i in range(4):
        # making separate first name column from new data frame
        if norm:
            new[i] = new[i].astype(int)/255.0
        df[col_name+"_%d"%i]= new[i]


    # Dropping old Name columns
    df.drop(columns =[col_name], inplace = True)
    return df

def hms_to_second(te_str):
    if isinstance(te_str, int):
        return te_str * 3600 / 86400
    if len(te_str)>10:
        datetime_object = datetime.strptime(te_str, "%Y-%m-%d %H:%M:%S")
    else:
        datetime_object = datetime.strptime(te_str, "%H:%M:%S")
    seconds = float(datetime_object.hour * 3600.0 + datetime_object.minute * 60.0 + datetime_object.second) / 86400.0
    print(te_str, datetime_object, seconds)
    return seconds


