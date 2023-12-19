import re
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Union, List, Tuple, Optional
import psycopg2
from psycopg2.extensions import connection
from tqdm import tqdm

DB_PARAMS: Dict[str, Union[str, int]] = {
    'dbname': 'aact',
    'user': 'wesserg',
    'password': 'h5p1le4sq',
    'host': 'aact-db.ctti-clinicaltrials.org',
    'port': 5432  # Default PostgreSQL port
}


def get_available_tables(db_params: Dict[str, Union[str, int]] = DB_PARAMS, aact_schema:str = 'ctgov') -> List[str]:
    """
    Retrieves a list of available tables in the AACT database.

    Args:
    - db_params (dict): Dictionary containing database connection parameters.

    Returns:
    - list: List of available tables in the AACT database.
    """
    conn = get_aact_connection(db_params)
    cursor = conn.cursor()
    aact_query = f"""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = '{aact_schema}';
    """
    cursor.execute(aact_query)
    tables = cursor.fetchall()
    conn.close()
    return tables


def get_aact_connection(db_params: Dict[str, Union[str, int]] = DB_PARAMS) -> connection:
    """
    Establishes a connection to the AACT database.

    Args:
    - db_params (dict): Dictionary containing database connection parameters.

    Returns:
    - connection: Psycopg2 database connection object.
    """
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        print("Connected to the database!")
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error:", error)


def get_aact_selected_data(aact_tables: List = ['studies'], aact_schema: str = 'ctgov', n_rows_limit: int = 1000) -> pd.DataFrame:
    """
    Retrieves data from the AACT database for a specified table and schema.

    Args:
    - aact_tables (list): The list of table names in the AACT database.
    - aact_schema (str): The schema name in the AACT database.
    - n_rows_limit (int): Maximum number of rows to fetch.

    Returns:
    - pd.DataFrame: DataFrame containing the queried data.
    """
    print("getting studies baseline data...")
    df = get_aact_ct_data(n_rows_limit=n_rows_limit)[0]
    nct_ids = list(df.nct_id.unique())
    nct_ids_str = ','.join(map(lambda x: f"'{x}'", nct_ids))
    df.set_index('nct_id', inplace=True)
    
    df_list = list()
    df_list.append(df)
    conn = get_aact_connection()
    cursor = conn.cursor()
    # Generate the SELECT statement for joining tables dynamically
    for table_name in tqdm(aact_tables):
        print(f"Fetching data from {table_name}...")
        aact_table_query = f"""
        SELECT * FROM {aact_schema}.{table_name} WHERE nct_id IN ({nct_ids_str});
        """
        cursor = conn.cursor()
        cursor.execute(aact_table_query)
        result = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

        # Convert the result to a DataFrame with column names
        df = pd.DataFrame(result, columns=column_names)
        df.drop("id", axis=1, inplace=True)
        if len(df) > 0 and "nct_id" in df.columns and df["nct_id"].nunique() == len(df):
            df.set_index('nct_id', inplace=True)
        elif len(df) > 0 and "nct_id" in df.columns and df["nct_id"].nunique() != len(df):
        
            grouped_cols = df.columns.to_series().groupby(df.dtypes.astype(str)).groups

            # Separate columns based on data types
            numeric_cols = [col for col in grouped_cols.get('float64', []) + grouped_cols.get('int64', [])]
            text_cols = [col for col in grouped_cols.get('object', []) if col != 'nct_id']

            # Aggregate numeric columns by taking the mean
            df_numeric_aggregated = df.groupby('nct_id')[numeric_cols].mean().reset_index()

            # Aggregate text columns by concatenating text values separated by comma
            df_text_aggregated = df.groupby('nct_id')[text_cols].agg(lambda x: ', '.join(x)).reset_index()

            # Merge the aggregated dataframes on 'nct_id'
            df = pd.merge(df_numeric_aggregated, df_text_aggregated, on='nct_id', how='outer')
            df.set_index('nct_id', inplace=True)
            print(df)

        else:
            print(table_name, "is empty!")
            continue
        if "id" in df.columns:
            df.drop('id', axis=1, inplace=True)
        df.rename({"name": table_name, "names": table_name}, axis=1, inplace=True)
        #df.drop(mostly_empty_columns+gibberish_columns, axis=1, inplace=True)
        #df_list.append(df[categorical_columns + numerical_columns + text_columns])    
        df.columns = [table_name + "_" + col for col in df.columns]
        df_list.append(df)    
    df = pd.concat(df_list, axis=1)
    text_columns = []
    categorical_columns = []
    numerical_columns = []
    gibberish_columns = []
    mostly_empty_columns = []
    for column in tqdm(df.columns):
        if column_is_empty(df[column]):
            mostly_empty_columns.append(column)
        elif column_is_categorical(df[column]):  # Detect categorical columns
            categorical_columns.append(column)
        elif column_is_numeric(df[column]):  # Detect numerical columns
            numerical_columns.append(column)
        elif column_contains_text(df[column]):  # Detect text columns
            text_columns.append(column)
        else:
            gibberish_columns.append(column)
    
    columns_dict = {'text_columns': text_columns, 'categorical_columns': categorical_columns,
                    'numerical_columns': numerical_columns, 'gibberish_columns': gibberish_columns,
                    'mostly_empty_columns': mostly_empty_columns}
    conn.close()    
    return df, columns_dict


def get_aact_ct_data(aact_table: str = 'studies', aact_schema: str = 'ctgov', n_rows_limit: int = 1000) -> pd.DataFrame:
    """
    Retrieves data from the AACT database for a specified table and schema.

    Args:
    - aact_table (str): The table name in the AACT database.
    - aact_schema (str): The schema name in the AACT database.
    - n_rows_limit (int): Maximum number of rows to fetch.

    Returns:
    - pd.DataFrame: DataFrame containing the queried data.
    """
    conn = get_aact_connection()
    cursor = conn.cursor()
    aact_query = f"""
    SELECT * FROM {aact_schema}.{aact_table}
    ORDER BY RANDOM()
    LIMIT {n_rows_limit};
    """

    cursor.execute(aact_query)
    
    result = cursor.fetchall()
    
    
    # Get column names
    column_names = [desc[0] for desc in cursor.description]

    # Convert the result to a DataFrame with column names
    df = pd.DataFrame(result, columns=column_names)
    print("Fetched the data!")
    print("Filtering variables by type...")
    text_columns = []
    categorical_columns = []
    numerical_columns = []
    gibberish_columns = []
    mostly_empty_columns = []
    for column in tqdm(df.columns):
        if column_is_empty(df[column]):
            mostly_empty_columns.append(column)
        elif column_is_categorical(df[column]):  # Detect categorical columns
            categorical_columns.append(column)
        elif column_is_numeric(df[column]):  # Detect numerical columns
            numerical_columns.append(column)
        elif column_contains_text(df[column]):  # Detect text columns
            text_columns.append(column)
        else:
            gibberish_columns.append(column)
    
    columns_dict = {'text_columns': text_columns, 'categorical_columns': categorical_columns,
                    'numerical_columns': numerical_columns, 'gibberish_columns': gibberish_columns,
                    'mostly_empty_columns': mostly_empty_columns}
    conn.close()
    return df, columns_dict


def get_aact_all_data(aact_schema: str = 'ctgov', n_rows_limit: int = 1000) -> pd.DataFrame:
    
    tables = get_available_tables()
    # %%
    # Analyze missing data for each column in a table
    all_tables_dict = dict()
    df_list = list()
    all_text_columns = []
    all_categorical_columns = []
    all_numerical_columns = []
    all_gibberish_columns = []
    all_mostly_empty_columns = []
    conn = get_aact_connection()
    for table_name in tqdm(tables):
        aact_table_query = f"""SELECT * FROM {aact_schema}.{table_name[0]} LIMIT {n_rows_limit};"""
        cursor = conn.cursor()
        cursor.execute(aact_table_query)
        result = cursor.fetchall()
        column_names = [table_name[0] + "_" + desc[0] for desc in cursor.description]

        # Convert the result to a DataFrame with column names
        df = pd.DataFrame(result, columns=column_names)
        if len(df) > 0 and "nct_id" in df.columns and df["nct_id"].nunique() == len(df):
            df.set_index('nct_id', inplace=True)
        else:
            continue
        if "id" in df.columns:
            df.drop('id', axis=1, inplace=True)
        df.rename({"name": table_name[0], "names": table_name[0]}, axis=1, inplace=True)
        text_columns = []
        categorical_columns = []
        numerical_columns = []
        gibberish_columns = []
        mostly_empty_columns = []
        for column in df.columns:
            if column_is_empty(df[column]):
                mostly_empty_columns.append(column)
            elif column_is_categorical(df[column]):  # Detect categorical columns
                categorical_columns.append(column)
            elif column_is_numeric(df[column]):  # Detect numerical columns
                numerical_columns.append(column)
            elif column_contains_text(df[column]):  # Detect text columns
                text_columns.append(column)
            else:
                gibberish_columns.append(column)
        all_mostly_empty_columns = mostly_empty_columns.copy()
        all_text_columns += text_columns.copy()
        all_categorical_columns += categorical_columns.copy()
        all_numerical_columns += numerical_columns.copy()
        all_gibberish_columns += gibberish_columns.copy()

        all_tables_dict[table_name[0]] = {'text_columns': text_columns.copy(), 'categorical_columns': categorical_columns.copy(),
                                            'numerical_columns': numerical_columns.copy(), 'gibberish_columns': gibberish_columns.copy(),
                                            'mostly_empty_columns': mostly_empty_columns.copy()}
        df.drop(mostly_empty_columns+gibberish_columns, axis=1, inplace=True)
        df_list.append(df[categorical_columns + numerical_columns + text_columns])    

    # %%
    all_tables_df = pd.concat([add_df for add_df in df_list if len(add_df)>0], axis=1)
    all_tables_df.dropna(axis=0, thresh=int(all_tables_df.shape[1]/5), inplace=True)
    all_tables_df.dropna(axis=1, thresh=int(all_tables_df.shape[0]/5), inplace=True)
    all_categorical_columns = np.intersect1d(all_categorical_columns, all_tables_df.columns)
    all_numerical_columns = np.intersect1d(all_numerical_columns, all_tables_df.columns)
    all_text_columns = np.intersect1d(all_text_columns, all_tables_df.columns)
    all_gibberish_columns = np.intersect1d(all_gibberish_columns, all_tables_df.columns)
    all_mostly_empty_columns = np.intersect1d(all_mostly_empty_columns, all_tables_df.columns)
    
    columns_dict = {'text_columns': all_text_columns, 'categorical_columns': all_categorical_columns,
                    'numerical_columns': all_numerical_columns, 'gibberish_columns': all_gibberish_columns,
                    'mostly_empty_columns': all_mostly_empty_columns}
    conn.close()    
    return df, columns_dict


def column_is_empty(data_column: pd.Series, min_empty_percent=90) -> bool:
    """
    Check if a DataFrame column is empty in min_empty_percent or more entries.

    Args:
        data_column (pd.Series): The pandas Series representing the column.
        min_empty_percent (int, optional): Minimum percentage of empty values to consider as empty. Defaults to 50.
    Returns:
        bool: True if the column is empty, otherwise False.
    """
    # Check if the column is empty
    if data_column.isnull().sum() / len(data_column) * 100 > min_empty_percent:
        return True
    else:
        return False

    
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


def column_is_categorical(data_column: pd.Series, max_unique_values: int = 100) -> bool:
    """
    Check if a DataFrame column is categorical.

    Args:
        data_column (pd.Series): The pandas Series representing the column.
        max_unique_values (int, optional): Maximum unique values to consider as categorical. Defaults to 100.

    Returns:
        bool: True if the column is categorical, otherwise False.
    """
    return data_column.nunique() < max_unique_values


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