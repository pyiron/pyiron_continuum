import difflib
from mendeleev.fetch import fetch_table


def get_tag(tag, arr, cutoff=0.8):
    results = difflib.get_close_matches(tag, arr, cutoff=cutoff)
    if len(results) == 0:
        raise KeyError(f"'{tag}' not found")
    return results[0]


def get_atom_info(name, column_in, column_out, difflib_cutoff=0.8):
    """
    Get atomic information from the periodic table.

    Parameters
    ----------
    name : str
        Element name or symbol.
    column_in : str
        Column name to search for the element.
    column_out : str
        Column name to return.
    difflib_cutoff : float, optional
        Cutoff for difflib.get_close_matches. Default is 0.8.

    Returns
    -------
    str
        Atomic information.
    """
    df = fetch_table("elements")
    if difflib_cutoff < 1:
        name = get_tag(name, df[column_in], cutoff=difflib_cutoff)
    data = df[df[column_in] == name]
    if len(data) == 0:
        raise KeyError(f"'{name}' not found")
    if len(data) > 1:
        raise KeyError(f"'{name}' is not uniquely defined")
    return list(data[column_out])[0]
