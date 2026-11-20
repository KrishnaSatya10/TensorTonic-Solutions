import pandas as pd

def multi_agg(data: dict, group_col: str, value_col: str, funcs: list) -> dict:
    """
    Returns a dictionary from function names to dictionaries of group aggregates.
    """
    df = pd.DataFrame(data)

    res = df.groupby(group_col)[value_col].agg(funcs)
    print(res)
    return res.to_dict()