import pandas as pd

def boolean_filter(data: dict, column: str, threshold: float) -> dict:
    """
    Returns filtered_data as a dictionary of lists and count as an integer.
    """
    df = pd.DataFrame(data)
    filt = df[df[column] > threshold]
    return {"filtered_data": filt.to_dict("list"),
            "count": filt.shape[0] }
