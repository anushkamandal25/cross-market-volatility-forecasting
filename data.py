import sqlite3
import pandas as pd
import requests
from config import settings


class TwelveDataAPI:
    def __init__(self, api_key=settings.twelvedata_api_key):
        self.__api_key = api_key

    def get_daily(self, ticker, outputsize=3000):
        """Get daily time series of an equity from TwelveData API.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the equity.
        outputsize : int, optional
            Number of observations to retrieve. By default 3000.

        Returns
        -------
        pd.DataFrame
            Columns are 'open', 'high', 'low', 'close', and 'volume'.
            All columns are numeric. Index is DatetimeIndex named 'date',
            sorted descending (most recent first).
        """
        url = (
            "https://api.twelvedata.com/time_series?"
            f"symbol={ticker}&"
            "interval=1day&"
            f"outputsize={outputsize}&"
            "format=JSON&"
            f"apikey={self.__api_key}"
        )

        response = requests.get(url=url)
        response_data = response.json()

        # Check for errors
        if "values" not in response_data:
            raise Exception(
                f"Invalid API call. Check that ticker symbol '{ticker}' is correct. "
                f"API message: {response_data.get('message', 'Unknown error')}"
            )

        # Convert to DataFrame
        df = pd.DataFrame(response_data["values"])

        # Set and convert index
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.index.name = "date"

        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]]

        # Convert to numeric
        df = df.astype(float)

        # Sort descending (most recent first) - consistent with original code
        df.sort_index(ascending=False, inplace=True)

        return df


class SQLRepository:
    def __init__(self, connection):
        self.connection = connection

    def insert_table(self, table_name, records, if_exists="fail"):
        """Insert DataFrame into SQLite database as table.

        Parameters
        ----------
        table_name : str
        records : pd.DataFrame
        if_exists : str, optional
            'fail', 'replace', or 'append'. Default: 'fail'

        Returns
        -------
        dict
            - 'transaction_successful': bool
            - 'records_inserted': int
        """
        n_inserted = records.to_sql(
            name=table_name, con=self.connection, if_exists=if_exists
        )

        return {
            "transaction_successful": True,
            "records_inserted": n_inserted
        }

    def read_table(self, table_name, limit=None):
        """Read table from database.

        Parameters
        ----------
        table_name : str
            Name of table in SQLite database.
        limit : int, None, optional
            Number of most recent records to retrieve. If None, all
            records are retrieved. By default None.

        Returns
        -------
        pd.DataFrame
            Index is DatetimeIndex 'date'. Columns are 'open', 'high',
            'low', 'close', and 'volume'. All columns are numeric.
        """
        if limit:
            sql = f"SELECT * FROM '{table_name}' LIMIT {limit}"
        else:
            sql = f"SELECT * FROM '{table_name}'"

        df = pd.read_sql(
            sql=sql, con=self.connection, parse_dates=["date"], index_col="date"
        )

        return df