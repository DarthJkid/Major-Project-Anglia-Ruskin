"""
fbref_scraper.py
==================

This module provides helper functions and a simple command‑line interface for
retrieving player performance data from the football statistics site FBref.

The underlying HTML on FBref pages often nests statistical tables inside HTML
comments to deter simple scraping methods.  In addition, FBref deploys
anti‑scraping measures such as Cloudflare that can return HTTP 403 errors
when a request does not appear to come from a real browser.  To overcome
these obstacles, this scraper uses the `beautifulsoup4` library to parse
tables hidden in comments and can optionally rely on the `cloudscraper`
package to mimic a standard browser request.  These steps follow the
approach described by Siddhraj Thakor, who notes that FBref tables are
hidden within HTML comments and that Cloudflare protection can block
basic HTTP requests【882069327944702†L59-L63】.  Using a library like
`cloudscraper` helps sidestep 403 errors by emulating a browser【882069327944702†L72-L96】.  When
scraping, always respect the site's terms of use and avoid overloading
their servers【882069327944702†L170-L172】.

The primary entry point is the `fetch_fbref_tables` function, which
downloads the provided FBref stats page and extracts every statistical
table found within ``<div>`` elements whose IDs begin with ``all_``.
Each table is returned as a pandas DataFrame with flattened column
headers.  You can also request a specific table by ID (for example,
``stats_standard`` for the Player Standard Stats).

Example
-------

>>> from fbref_scraper import fetch_fbref_tables
>>> tables = fetch_fbref_tables(
...     "https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats",
...     table_id="stats_standard",
... )
>>> df = tables["stats_standard"]
>>> print(df.head())

Command‑line usage
------------------

```
python fbref_scraper.py --url <FBREF_STATS_URL> [--table <TABLE_ID>] [--output <DIRECTORY>]
```

If ``--table`` is omitted, the scraper will download all available tables on
the page.  If ``--output`` is provided, each extracted table will be
written to a CSV file in the specified directory.

Dependencies
------------

* pandas
* beautifulsoup4
* requests (standard library requests) or cloudscraper (optional)

To install the optional dependency ``cloudscraper`` for handling
Cloudflare's anti‑bot protections, run::

    pip install cloudscraper

```
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

try:
    import cloudscraper  # type: ignore
except ImportError:
    cloudscraper = None  # type: ignore


def _create_session(use_cloudscraper: bool = True) -> requests.Session:
    """Create an HTTP session.

    If ``use_cloudscraper`` is True and the optional ``cloudscraper`` package
    is available, a Cloudflare‑aware scraper is returned.  Otherwise, a
    vanilla ``requests.Session`` is created.

    Parameters
    ----------
    use_cloudscraper : bool
        Whether to use ``cloudscraper`` when available.  If the
        ``cloudscraper`` module is not installed, a normal session is used
        regardless of this flag.

    Returns
    -------
    requests.Session
        An HTTP session object suitable for performing GET requests.
    """
    if use_cloudscraper and cloudscraper is not None:
        # Create a Cloudflare‑aware scraper.  This mimics a regular browser
        # request and can reduce the likelihood of encountering a 403 error【882069327944702†L59-L63】.
        return cloudscraper.create_scraper()  # type: ignore[no-any-return]
    # Fallback to a standard requests Session.
    return requests.Session()


def _fetch_page(session: requests.Session, url: str) -> str:
    """Retrieve the HTML content of a web page.

    Parameters
    ----------
    session : requests.Session
        Session used to send the GET request.
    url : str
        URL of the page to download.

    Returns
    -------
    str
        The response text (HTML) of the page.

    Raises
    ------
    requests.HTTPError
        If the request returns a non‑200 HTTP status code.
    """
    response = session.get(url)
    response.raise_for_status()
    return response.text


def _flatten_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a DataFrame's multi‑index columns into single strings.

    FBref tables often have multi‑level column headers representing
    hierarchical information (e.g., a first level of group names and a
    second level of statistic names).  To simplify further analysis, this
    helper joins the tuples into single strings with spaces.  If a column
    header is already a string, it is left unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns will be flattened.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with flattened column names.
    """
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            # Join tuple components and strip whitespace
            new_columns.append(" ".join([str(c).strip() for c in col if c]))
        else:
            new_columns.append(str(col).strip())
    df.columns = new_columns
    return df


def _extract_table_from_div(div: BeautifulSoup, table_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Parse a single FBref table hidden inside an HTML comment.

    FBref wraps many of its tables in an HTML comment inside a ``div`` with
    an ID like ``all_stats_standard``.  This function extracts the comment,
    parses it, and reads the embedded ``<table>`` into a pandas
    ``DataFrame``.  The resulting DataFrame has flattened column headers.

    Parameters
    ----------
    div : BeautifulSoup
        The ``div`` element whose comment contains the table.
    table_id : Optional[str], optional
        ID of the table to extract.  If provided, the function only
        returns data for a table whose ``<table>`` element has a matching
        ID.  When ``None``, the first table in the comment is returned.

    Returns
    -------
    Optional[pandas.DataFrame]
        A DataFrame containing the parsed table, or ``None`` if no
        matching table is found.
    """
    # The table is stored within an HTML comment.
    comment = div.find(string=lambda text: isinstance(text, Comment))
    if not comment:
        return None
    # Parse the comment's HTML
    soup = BeautifulSoup(comment, "html.parser")
    # If a specific table ID is requested, locate it; otherwise, pick the first table.
    table_tag = soup.find("table", id=table_id) if table_id else soup.find("table")
    if table_tag is None:
        return None
    # Use pandas to read the table into a DataFrame.  read_html returns a list.
    try:
        df = pd.read_html(str(table_tag))[0]
    except ValueError:
        # read_html failed – return None to signal an empty table.
        return None
    return _flatten_headers(df)


def fetch_fbref_tables(
    url: str,
    table_id: Optional[str] = None,
    use_cloudscraper: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Download and parse statistical tables from a FBref stats page.

    This function fetches the HTML content from the supplied FBref URL,
    identifies all ``<div>`` containers whose ``id`` attribute begins with
    ``"all_"`` (which is the convention FBref uses for wrapping tables),
    extracts each table hidden inside the nested HTML comment, and returns
    the results as a dictionary mapping table identifiers to pandas
    DataFrames.  If ``table_id`` is specified, only that particular table
    is extracted (useful when you know the exact ID, such as
    ``"stats_standard"`` for Player Standard Stats).

    Parameters
    ----------
    url : str
        Full URL to the FBref stats page.  For example:
        ``"https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats"``.
    table_id : Optional[str], default ``None``
        ID of a specific table to extract.  If provided, only one entry
        corresponding to this ID will appear in the returned dictionary.
    use_cloudscraper : bool, default ``True``
        When ``True`` and the optional ``cloudscraper`` package is
        installed, the scraper will attempt to mimic a browser to reduce
        the chance of HTTP 403 responses.  Set this to ``False`` to use a
        plain ``requests`` session.

    Returns
    -------
    Dict[str, pandas.DataFrame]
        A mapping from table identifiers (e.g., ``"stats_standard"``) to
        DataFrames containing the parsed data.  If no tables are found,
        an empty dictionary is returned.
    """
    session = _create_session(use_cloudscraper=use_cloudscraper)
    html = _fetch_page(session, url)
    soup = BeautifulSoup(html, "html.parser")
    tables: Dict[str, pd.DataFrame] = {}
    # Regular expression to match IDs that start with "all_"
    pattern = re.compile(r"^all_")
    # Iterate through all divs and extract tables
    for div in soup.find_all("div", id=pattern):
        div_id = div.get("id", "")
        # Derive the table identifier by stripping the "all_" prefix
        key = div_id[4:]
        # If a specific table_id is requested and this one does not match, skip it
        if table_id is not None and key != table_id:
            continue
        df = _extract_table_from_div(div, table_id=key)
        if df is not None:
            tables[key] = df
    return tables


def save_tables_to_csv(tables: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Save DataFrames to CSV files in the specified directory.

    Each table in the ``tables`` dictionary will be written to
    ``<output_dir>/<table_name>.csv``.  The output directory is created
    if it does not already exist.

    Parameters
    ----------
    tables : Dict[str, pandas.DataFrame]
        Mapping from table names to DataFrames.
    output_dir : str
        Directory path in which CSV files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, df in tables.items():
        # Construct a safe file name
        fname = f"{name}.csv"
        path = os.path.join(output_dir, fname)
        df.to_csv(path, index=False)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for command‑line invocation.

    Parses arguments and orchestrates fetching and saving of FBref tables.

    Parameters
    ----------
    argv : Optional[list[str]]
        List of command‑line arguments.  If ``None``, ``sys.argv[1:]`` is used.

    Returns
    -------
    int
        Exit status code (0 on success, non‑zero on failure).
    """
    parser = argparse.ArgumentParser(description="Download tables from an FBref stats page.")
    parser.add_argument(
        "--url",
        required=True,
        help="FBref stats page URL to scrape (e.g., https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats)",
    )
    parser.add_argument(
        "--table",
        default=None,
        help=(
            "Specific table ID to extract (e.g., 'stats_standard').  If omitted, all tables on the page are returned."
        ),
    )
    parser.add_argument(
        "--no-cloudscraper",
        action="store_true",
        help="Do not use cloudscraper even if it is installed; fall back to plain requests.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=("Optional directory path to save extracted tables as CSV files.  If not provided, tables are printed to stdout."),
    )
    args = parser.parse_args(argv)

    # Determine whether to use cloudscraper
    use_cloud = not args.no_cloudscraper
    try:
        tables = fetch_fbref_tables(url=args.url, table_id=args.table, use_cloudscraper=use_cloud)
    except Exception as exc:
        sys.stderr.write(f"Error fetching tables: {exc}\n")
        return 1
    if not tables:
        sys.stderr.write("No tables were found or extracted.\n")
        return 1
    if args.output:
        save_tables_to_csv(tables, args.output)
    else:
        for name, df in tables.items():
            print(f"\nTable: {name}")
            print(df.head())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())