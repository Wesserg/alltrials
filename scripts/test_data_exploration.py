# %% [markdown]
# # Exploring the Cinical Trials dataset
# %% 
# ## Importing libraries
import pandas as pd
from datetime import datetime as dt
import re
from tqdm import tqdm
import numpy as np
reload_all_data = False
# %% loading the large pandas dataset
# wonder if we can read it in into memory:P - YES we can.
if reload_all_data:
    print(f"{dt.now()}: Reading in csv data...")
    alltrials_df = pd.read_csv('../data/alltrials.csv', index_col=False, on_bad_lines='warn', sep="\t")
    print(f"{dt.now()}: Done reading in csv data...")

# %% 
# ## Exploring the data
if reload_all_data:

    n,m = alltrials_df.shape

    # get all columns in the dataframe
    useful_columns = alltrials_df.columns
    print(len(useful_columns))
    # How many columns have what percentage of missing values?
    percent_null_columns = alltrials_df.isnull().sum(axis=0)/n
    # Remove columns from useful_columns with more than npercent_threshold missing values
    npercent_threshold = 0.80
    useful_columns = useful_columns[percent_null_columns < npercent_threshold]
    print(len(useful_columns))

    # remove columns with only 1 unique value except nan
    unique_columns = alltrials_df[useful_columns].nunique()
    unique_columns = unique_columns[unique_columns > 1]
    useful_columns = unique_columns.index
    print(len(useful_columns))
# %%
# Restrict dataframe to useful columns
if reload_all_data:

    alltrials_df = alltrials_df[useful_columns]
    print(alltrials_df.shape)

# %%
# Remove trials (rows of the dataframe) that have more than npercent_threshold missing values
if reload_all_data:

    npercent_threshold = 0.5
    percent_null_rows = alltrials_df.isnull().sum(axis=1)/m
    alltrials_df = alltrials_df.loc[(percent_null_rows < npercent_threshold),:]
    alltrials_df.to_csv("../data/alltrials_clean.csv", index=False, sep="\t")

# %%
# %% Load the cleaned data
alltrials_df = pd.read_csv('../data/alltrials_clean.csv', index_col=False, on_bad_lines='warn', sep="\t")
alltrials_df.shape
# %% [markdown]
# 

from sklearn.preprocessing import LabelEncoder

# Sample clinical trial data (replace with your dataframe)
alltrials_df = alltrials_df.sample(10000)
data = alltrials_df
data_types = data.dtypes


def column_contains_text(data_column, min_text_len = 100, min_unique_values=20):
    # Iterate through the values in the column
    for value in data_column:
        if data_column.nunique() > min_unique_values: 
            if (isinstance(value, str)) and (re.search(r'\w+', value)) and (len(value) > min_text_len):
                return True  # Text detected in at least one value
    return False  # No text detected in any value


def column_is_numeric(data_column, min_unique_values=20):
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


text_columns = []
categorical_columns = []
numerical_columns = []
gibberish_columns = []

# %%
# Categorize columns based on data types
# iterate throuhg columns and data types

for column, data_type in tqdm(data_types.items()):
    if column_contains_text(data[column]):  # Detect text columns
        text_columns.append(column)
    elif column_is_numeric(data[column]):  # Detect numerical columns
        numerical_columns.append(column)
    elif column_is_categorical(data[column]):  # Detect categorical columns
        categorical_columns.append(column)
    else:
        gibberish_columns.append(column)

print("Text Columns:", text_columns)
print("Categorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)
print("Gibberish Columns:", gibberish_columns)
# Extract numerical features and handle missing values
# For this example, we will keep 'TrialID' as a numerical feature
# %% [optional]
# Using TF-IDF to convert text columns into numerical features 
tfidf_vectorizer = TfidfVectorizer()
text_embeddings = tfidf_vectorizer.fit_transform(data[text_columns])

# Convert categorical variables into numerical representations
categorical_features = ['Condition']
label_encoders = {}
for feature in categorical_columns:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# Combine all numerical features
numerical_data = data[numerical_columns + categorical_columns]

# Combine the numerical data with text embeddings
combined_data = pd.concat([numerical_data, pd.DataFrame(text_embeddings.toarray())], axis=1)

# Print the resulting dataframe
print(combined_data)


# %% [Averaging word vectors for a sentence/paragraph]
# Using Word2Vec to convert text columns into embeddings
# This is not the most efficient as it is not generalizable to words that have not been seen before
from gensim.models import Word2Vec, fasttext, Doc2Vec #FB alternative to word2vec
data[text_columns] = data[text_columns].fillna(' ')  # Replace missing values with empty strings
data['ConcatenatedText'] = data[text_columns].apply(lambda row: '. '.join(row), axis=1)
symbols_to_remove = r'[\[\]{}]'  # This regex pattern removes square brackets and curly braces
# Remove symbols indicating lists or sets
data['ConcatenatedText'] = data['ConcatenatedText'].apply(lambda text: re.sub(symbols_to_remove, '', text))
# %%
text_data = data['ConcatenatedText']
model = Word2Vec(sentences=text_data, vector_size=100, window=5, min_count=1, sg=0)
vector = model.wv['a']
# Example: Averaging word vectors for a sentence
# %%
sentence = ['This', 'is', 'a', 'sentence']
sentence_vector = np.mean([model.wv[word] for word in sentence], axis=0)


# %% [Using Doc2Vec to convert text into embeddings]
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# Assuming 'data' is your DataFrame with the 'ConcatenatedText' column
documents = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(data['ConcatenatedText'])]

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
