
# Using Doc2Vec to convert text columns into embeddings

import pandas as pd
import re
import numpy as np
from gensim.models import Doc2Vec # Word2Vec, fasttext, 
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import StandardScaler


def get_text_embeddings(alltrials_text_df: pd.DataFrame, center_scaling: bool = True) -> pd.DataFrame:
    alltrials_text_df.replace(" ", np.nan, inplace=True)
    
    alltrials_text_df['Notes'] = alltrials_text_df.apply(lambda row: '. '.join(filter(lambda y: isinstance(y, str), row)), axis=1)
    symbols_to_remove = r'[\|\[\]{}\n\r"\']'  # This regex pattern removes square brackets and curly braces
    # Remove symbols indicating lists or sets
    alltrials_text_df['Notes'] = alltrials_text_df['Notes'].apply(lambda text: re.sub(symbols_to_remove, ' ', text.lower()))
    documents = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(alltrials_text_df['Notes'])]

    # Train a Doc2Vec model
    model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4, epochs=20)
    # Get the document vector for a specific clinical trial

    if center_scaling:
        model_vectors = np.array([model.dv[i] for i in range(len(model.dv))])
        scaler = StandardScaler()
        scaler.fit(model_vectors)
        model_vectors_centerscaled = scaler.transform(model_vectors)

    return model_vectors_centerscaled
