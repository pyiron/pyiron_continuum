import difflib
from mendeleev.fetch import fetch_table


def get_tag(tag, arr, cutoff=0.8):
    results = difflib.get_close_matches(tag, arr, cutoff=cutoff)
    if len(results) == 0:
        raise KeyError(f"'{tag}' not found")
    return results[0]


def get_atom_info(difflib_cutoff=0.8, **kwargs):
    """
    Get atomic information from the periodic table.

    Args:
        difflib_cutoff (float): Cutoff for difflib.get_close_matches
        **kwargs: Key-value pairs to search for

    Returns:
        dict: Atomic information
    """
    df = fetch_table("elements")
    if len(kwargs) == 0:
        raise ValueError("No arguments provided")
    for key, tag in kwargs.items():
        if difflib_cutoff < 1:
            key = get_tag(key, df.keys(), cutoff=difflib_cutoff)
            tag = get_tag(tag, df[key], cutoff=difflib_cutoff)
            if sum(df[key] == tag) == 0:
                raise KeyError(f"'{tag}' not found")
            df = df[df[key] == tag]
    return df.squeeze(axis=0).to_dict()
