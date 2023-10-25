# %% [markdown]
# # Exploring the Cinical Trials dataset
# %% 
# ## Importing libraries
import pandas as pd
from datetime import datetime as dt
import re
from tqdm import tqdm
import numpy as np
import graph_tool.all as gt
import sqlite3
import psycopg2


reload_all_data = True
use_sqlite = True
use_csv = False
use_postgres = False
table_name = "alltrials"
data_dir = "../data/"
n_samples = 10000
# %% loading the large pandas dataset
# wonder if we can read it in into memory:P - YES we can.
print(f"{dt.now()}: Reading in data...")
# %%
def column_contains_text(data_column, min_text_len = 100, min_unique_values=20):
    # Iterate through the values in the column
    for value in data_column:
        if data_column.nunique() > min_unique_values: 
            if (isinstance(value, str)) and (re.search(r'\w+', value)) and (len(value) > min_text_len):
                return True  # Text detected in at least one value
    return False  # No text detected in any value


def column_is_numeric(data_column, min_unique_values=20):
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


def column_is_categorical(data_column, max_unique_values=100):
    return data_column.nunique() < max_unique_values  # Check if there are multiple unique values


def load_data(data_dir: str, table_name: str, n_samples: int =0, load_source: str = "sqlite", drop_empty_columns_threshold = 0.8, drop_empty_rows_threshold = 0.2) -> pd.DataFrame:
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

# %% 
# Testing the loading of the data
table_name = "alltrials"
data_dir = "../data/"
n_samples = 10000
load_source = "sqlite"
print(f"{dt.now()}: Reading in data from '{table_name} table in '{data_dir}' from {load_source}...")

alltrials_df = load_data(data_dir, table_name, n_samples, load_source=load_source)
print(f"{dt.now()}: Done reading in data...")

print(alltrials_df)
# %%
# Sample clinical trial data (replace with your dataframe)

data_types = alltrials_df.dtypes

# %%
# Group columns based on data types
# - categorical variables
# - text variables
# - numerical variables
# - gibberish variables (e.g. columns with only one value) or too many unique non-numeric values
text_columns = []
categorical_columns = []
numerical_columns = []
gibberish_columns = []
for column, data_type in tqdm(data_types.items()):
    if column_is_categorical(alltrials_df[column]):  # Detect categorical columns
        categorical_columns.append(column)
    elif column_is_numeric(alltrials_df[column]):  # Detect numerical columns
        numerical_columns.append(column)
    elif column_contains_text(alltrials_df[column]):  # Detect text columns
        text_columns.append(column)
    else:
        gibberish_columns.append(column)

print("Text Columns:", text_columns)
print("Categorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)
print("Gibberish Columns:", gibberish_columns)

# %% [Averaging word vectors for a sentence/paragraph]
# Using Word2Vec to convert text columns into embeddings
# This is not the most efficient as it is not generalizable to words that have not been seen before
from gensim.models import Doc2Vec # Word2Vec, fasttext, 
from gensim.models.doc2vec import TaggedDocument

alltrials_df[text_columns] = alltrials_df[text_columns].fillna(' ')  # Replace missing values with empty strings
alltrials_df['Notes'] = alltrials_df[text_columns].apply(lambda row: '. '.join(row), axis=1)
symbols_to_remove = r'[\[\]{}]'  # This regex pattern removes square brackets and curly braces
# Remove symbols indicating lists or sets
alltrials_df['Notes'] = alltrials_df['Notes'].apply(lambda text: re.sub(symbols_to_remove, '', text.lower()))
text_data = alltrials_df['Notes']


# %% [Using Doc2Vec to convert text into embeddings]

# Assuming 'data' is your DataFrame with the 'ConcatenatedText' column
documents = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(alltrials_df['Notes'])]

# Train a Doc2Vec model
model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4, epochs=20)

# Get the document vector for a specific clinical trial
vector = model.dv[0]  # i is the index of the clinical trial

# %%
new_text = "This is a new clinical trial description."
new_text_preprocessed = new_text.split()  # Replace with your preprocessing steps
new_tag = "new_trial"

# Create a TaggedDocument for the new text
new_document = TaggedDocument(words=new_text_preprocessed, tags=[new_tag])

# Infer the embedding for the new text
new_vector = model.infer_vector(new_document.words)

# %%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
i, j = 1,100
emb_i = model.dv[i]
emb_j = model.dv[j]
cosine_sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))

print(f"Cosine Similarity between trials {i} and {j}: {cosine_sim}")

# %%


# Cosine similarity threshold for adding edges
cosine_threshold = 0.7

# Create a graph
g = gt.Graph(directed=False)

# Add a property map to store clinical trial IDs as node properties
node_ids = g.new_vertex_property("int")

# Loop through clinical trials
for i, trial1 in enumerate(model.dv):
    v1 = g.add_vertex()  # Add a node for clinical trial 1
    node_ids[v1] = trial1["trial_id"]  # Set the node property to store trial ID

    for j, trial2 in enumerate(model.dv[i+1:]):
        if trial1["trial_id"] != trial2["trial_id"]:
            # Calculate cosine similarity between trial1 and trial2
            embedding1 = trial1["embedding"]
            embedding2 = trial2["embedding"]
            cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

            if cosine_similarity < cosine_threshold:
                # Add an edge between trial1 and trial2
                v2 = g.vertex(g.vertex_index[v1])  # Get vertex for trial1
                v3 = g.vertex(trial2["trial_id"])  # Get vertex for trial2
                g.add_edge(v2, v3)