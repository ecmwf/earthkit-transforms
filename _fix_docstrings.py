"""Replace abbreviated xarray/pandas/numpy type names with fully qualified names
inside docstrings only (leaves actual Python code untouched).
"""
import io
import tokenize

REPLACEMENTS = [
    ("xr.DataArray", "xarray.DataArray"),
    ("xr.Dataset", "xarray.Dataset"),
    ("pd.Timedelta", "pandas.Timedelta"),
    ("pd.DataFrame", "pandas.DataFrame"),
    ("np.ndarray", "numpy.ndarray"),
]

TARGET_FILES = [
    "src/earthkit/transforms/temporal/_aggregate.py",
    "src/earthkit/transforms/temporal/_rates.py",
    "src/earthkit/transforms/ensemble/_aggregate.py",
    "src/earthkit/transforms/climatology/_aggregate.py",
    "src/earthkit/transforms/spatial/_aggregate.py",
]


def apply_replacements(s):
    for old, new in REPLACEMENTS:
        s = s.replace(old, new)
    return s


for path in TARGET_FILES:
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    lines = source.splitlines(keepends=True)

    # Collect edits: (start_offset, end_offset, replacement)
    edits = []
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        raw = tok.string
        # Only process triple-quoted strings (docstrings)
        if not (raw.startswith('"""') or raw.startswith("'''")):
            continue
        new_raw = apply_replacements(raw)
        if new_raw == raw:
            continue
        start_row, start_col = tok.start
        end_row, end_col = tok.end
        start_off = sum(len(lines[i]) for i in range(start_row - 1)) + start_col
        end_off = sum(len(lines[i]) for i in range(end_row - 1)) + end_col
        edits.append((start_off, end_off, new_raw))

    if not edits:
        print(f"  (no changes) {path}")
        continue

    # Apply edits in reverse order to keep offsets valid
    new_source = source
    for start_off, end_off, new_raw in sorted(edits, reverse=True):
        new_source = new_source[:start_off] + new_raw + new_source[end_off:]

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_source)
    print(f"  updated {path}  ({len(edits)} docstring(s) changed)")

print("Done.")
