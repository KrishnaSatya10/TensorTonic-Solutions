import pandas as pd

def iloc_selection(data: dict, row: int, col: int) -> list:
    """
    Returns [element, row_values, col_values], with both value sequences as lists.
    """
    df = pd.DataFrame(data)
    elem = df.iloc[row,col]
    row_vals = df.iloc[row,:].tolist()
    col_vals = df.iloc[:,col].tolist()

    return [elem, row_vals, col_vals]
