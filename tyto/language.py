
def list2str(
    iterable: list,
    conjunction: str="and",
    oxford_comma: bool=False,
) -> str:
    list_of_strs = [str(item) for item in iterable]
    return (
        f"{', '.join(list_of_strs[:-1])}{', ' if oxford_comma else ' '}"
        f"{conjunction} {list_of_strs[-1]}"
    )