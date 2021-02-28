# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:13:16 2021

@author: femiogundare
"""



def extract_coords(df):
    """
    Returns the coordinates (x, y) where each patch is found in the whole mount sample.
    
    Args:
        df: dataframe
    """
    
    coord = df.path.str.rsplit("_", n=4, expand=True)
    coord = coord.drop([0, 1, 4], axis=1)
    coord = coord.rename({2: "x", 3: "y"}, axis=1)
    coord.loc[:, "x"] = coord.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)
    coord.loc[:, "y"] = coord.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)
    df.loc[:, "x"] = coord.x.values
    df.loc[:, "y"] = coord.y.values
    return df