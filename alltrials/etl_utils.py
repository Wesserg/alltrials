
import re
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Union

def column_contains_text(data_column: pd.Series, min_text_len: int = 100, min_unique_values: int = 20) -> bool:
    """
    Check if a DataFrame column contains text values.

    Args:
        data_column (pd.Series): The pandas Series representing the column.
        min_text_len (int, optional): Minimum text length to consider as containing text. Defaults to 100.
        min_unique_values (int, optional): Minimum unique values to consider as containing text. Defaults to 20.

    Returns:
        bool: True if text is detected in the column, otherwise False.
    """
    # Iterate through the values in the column
    for value in data_column:
        if data_column.nunique() > min_unique_values: 
            if (isinstance(value, str)) and (re.search(r'\w+', value)) and (len(value) > min_text_len):
                return True  # Text detected in at least one value
    return False  # No text detected in any value


def column_is_empty(data_column: pd.Series, max_frac_empty: float = 0.8) -> bool:
    """
    Check if a DataFrame column is mostly empty.

    Args:
        data_column (pd.Series): The pandas Series representing the column.
        max_frac_empty (float, optional): Maximum fraction of empty values to consider as empty. Defaults to 0.8.

    Returns:
        bool: True if the column is empty, otherwise False.
    """
    if data_column.isnull().sum() / len(data_column) > max_frac_empty:
        return True
    else:
        return False

def column_is_numeric(data_column: pd.Series, min_unique_values: int = 20) -> bool:
    """
    Check if a DataFrame column is numeric.

    Args:
        data_column (pd.Series): The pandas Series representing the column.
        min_unique_values (int, optional): Minimum unique values to consider as numeric. Defaults to 20.

    Returns:
        bool: True if the column is numeric, otherwise False.
    """
    if pd.api.types.is_numeric_dtype(data_column.dtype):
        return True
    else:
        try:
            pd.to_numeric(data_column, errors='raise')
            if data_column.nunique() > min_unique_values:  # Check if there are multiple unique values
                return True
            else:
                return False
        except (ValueError, TypeError):
            return False


def column_is_categorical(data_column: pd.Series, max_unique_values: int = 100, min_unique_values: int = 1) -> bool:
    """
    Check if a DataFrame column is categorical.

    Args:
        data_column (pd.Series): The pandas Series representing the column.
        max_unique_values (int, optional): Maximum unique values to consider as categorical. Defaults to 100.

    Returns:
        bool: True if the column is categorical, otherwise False.
    """
    nunique_values = data_column.nunique()
    return nunique_values < max_unique_values and nunique_values > min_unique_values


def load_data(data_dir: str, table_name: str, n_samples: int = 0, load_source: str = "sqlite",
              drop_empty_columns_threshold: float = 0.8, drop_empty_rows_threshold: float = 0.2) -> pd.DataFrame:
    """
    Load data from different sources (SQLite, CSV, or PostgreSQL) and preprocess it.

    Args:
        data_dir (str): The directory containing the data.
        table_name (str): The name of the table or file to load.
        n_samples (int, optional): Number of samples to load (0 for all). Defaults to 0.
        load_source (str, optional): Data source ("sqlite", "csv", or "postgres"). Defaults to "sqlite".
        drop_empty_columns_threshold (float, optional): Threshold for dropping columns with missing values. Defaults to 0.8.
        drop_empty_rows_threshold (float, optional): Threshold for dropping rows with missing values. Defaults to 0.2.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    if load_source=="sqlite":
        db_file = f"{data_dir}{table_name}.db"  # Replace with the path to your SQLite database

        # Connect to the SQLite database
        conn = sqlite3.connect(db_file)

        # Create a SQL query to sample 'n' rows from your table
        if n_samples==0:
            query = f"SELECT * FROM '{table_name}';"    
        else:
            query = f"SELECT * FROM '{table_name}' ORDER BY RANDOM() LIMIT {n_samples};"
        col_query = f"PRAGMA table_info('{table_name}');"
        columns_info = conn.execute(col_query).fetchall()
        column_names = [col[1] for col in columns_info]
        # Use Pandas to read the query result into a DataFrame
        alltrials_df = pd.read_sql_query(query, conn)
        alltrials_df.columns = column_names
        # Close the database connection
        conn.close()
        
    elif use_csv:
        alltrials_df = pd.read_csv(f"{data_dir}{table_name}.csv", index_col=False, on_bad_lines='warn', sep="\t")
        if n_samples > 0:
            alltrials_df = alltrials_df.sample(n_samples)
    elif use_postgres:
        
        # Set your connection parameters
        db_params = {
            'dbname': 'aact',
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost',
            'port': 5432  # Default PostgreSQL port
        }

        # Connect to the database
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        print("Connected to the database!")
        
        if n_samples==0:
            query = f"SELECT * FROM '{table_name}';"    
        else:
            query = f"SELECT * FROM '{table_name}' ORDER BY RANDOM() LIMIT {n_samples};"

        # You can now execute SQL queries
        cursor.execute("SELECT * FROM ctgov.conditions LIMIT 10") # What other tables are there? See cell below.
        result = cursor.fetchall()
        print(result)
    
        # Define the column names
        column_names = [desc[0] for desc in cursor.description]

        # Create a Pandas DataFrame
        df = pd.DataFrame(result, columns=column_names)
        print(df)


        # Don't forget to close the cursor and connection when done
        cursor.close()
        conn.close()
    # Remove columns with more than 80% missing values
    alltrials_df = alltrials_df.dropna(axis=1, thresh=int(drop_empty_columns_threshold* len(alltrials_df)), inplace=False)
    # Remove rows with more than 20% missing values
    alltrials_df = alltrials_df.dropna(axis=0, thresh=int(drop_empty_rows_threshold* len(alltrials_df.columns)), inplace=False)
    return alltrials_df