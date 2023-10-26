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
from sklearn.metrics.pairwise import cosine_similarity

import sqlite3
import psycopg2
from alltrials.etl_utils import load_data, column_contains_text, column_is_numeric, column_is_categorical

reload_all_data = True
use_sqlite = True
use_csv = False
use_postgres = False
table_name = "alltrials"
data_dir = "../data/"
n_samples = 10000

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


# %%
# Group columns based on data types
# - categorical variables
# - text variables
# - numerical variables
# - gibberish variables (e.g. columns with only one value) or too many unique non-numeric values
data_types = alltrials_df.dtypes

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
symbols_to_remove = r'[\[\]{}\n\r"\']'  # This regex pattern removes square brackets and curly braces
# Remove symbols indicating lists or sets
alltrials_df['Notes'] = alltrials_df['Notes'].apply(lambda text: re.sub(symbols_to_remove, ' ', text.lower()))
text_data = alltrials_df['Notes']


# %% [Using Doc2Vec to convert text into embeddings]

# Assuming 'data' is your DataFrame with the 'ConcatenatedText' column
documents = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(alltrials_df['Notes'])]

# Train a Doc2Vec model
model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4, epochs=20)

# Get the document vector for a specific clinical trial
trial_embedding = model.dv[0]  # i is the index of the clinical trial

# %%
new_text = "This is a new clinical trial description."
new_text_preprocessed = new_text.split()  # Replace with your preprocessing steps
new_tag = "new_trial"

# Create a TaggedDocument for the new text
new_document = TaggedDocument(words=new_text_preprocessed, tags=[new_tag])

# Infer the embedding for the new text
new_trial_embedding = model.infer_vector(new_document.words)

# %%
emb_i = trial_embedding
emb_j = new_trial_embedding
cosine_sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))

print(f"Cosine Similarity between known trial and new trial: {cosine_sim}")

# %%


# Cosine similarity threshold for adding edges
cosine_threshold = 0.7

# Create a graph
g = gt.Graph(directed=False)

# Add a property map to store clinical trial IDs as node properties
node_ids = g.new_vertex_property("int")

# Loop through clinical trials
for i, trial1 in enumerate(tqdm(model.dv)):
    v1 = g.add_vertex()  # Add a node for clinical trial 1
    node_ids[v1] = i  # Set the node property to store trial ID

    for j in range(i + 1, len(model.dv)):
        v2 = g.add_vertex()  # Add a node for clinical trial 1
        node_ids[v2] = j
        # Calculate cosine similarity between trial1 and trial2
        embedding1 = trial1
        embedding2 = model.dv[j]
        cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

        if cosine_similarity < cosine_threshold:
            # Add an edge between trial1 and trial2
            g.add_edge(v1, v2)
# %%
print(g)
# Draw the graph using graph-tool
# Note: This will not work in JupyterLab
gt.graph_draw(g, vertex_text=node_ids, vertex_font_size=10, output_size=(1000, 1000), output="graph.png")
