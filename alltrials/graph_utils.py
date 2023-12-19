from sklearn.metrics.pairwise import cosine_similarity
import graph_tool.all as gt
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
    

def create_ct_graph(model_vectors_centerscaled, cosine_threshold=0.5):
    # Cosine similarity threshold for adding edges
    # Create a graph
    g = gt.Graph(directed=False)

    # Add a property map to store clinical trial IDs as node properties
    # Loop through clinical trials
    n_embeddings = len(model_vectors_centerscaled)
    edge_weights = []
    for i in tqdm(range(n_embeddings-1)):
        trial1 = model_vectors_centerscaled[i,:]
        #v1 = g.add_vertex()  # Add a node for clinical trial 1
        #node_ids[v1] = i  # Set the node property to store trial ID
        embeddings = []
        for doc_id in range(i+1, n_embeddings):
            embeddings.append(model_vectors_centerscaled[doc_id,:])  # Access embeddings from Doc2Vec model

    # Convert embeddings to a NumPy array
        embeddings_array = np.array(embeddings)
        cosine_similarity_vector = cosine_similarity(trial1.reshape(1, -1), embeddings_array)
        abs_cosine_similarity_vector = np.abs(cosine_similarity_vector)
        new_edge_weights = [cosine_similarity_vector[0][j]for j in range(i + 1, abs_cosine_similarity_vector.shape[1]) if abs_cosine_similarity_vector[0][j] > cosine_threshold]
        if len(new_edge_weights) > 0:
            edge_weights.append(new_edge_weights)
        edge_list = [(i, j) for j in range(i + 1, abs_cosine_similarity_vector.shape[1]) if abs_cosine_similarity_vector[0][j] > cosine_threshold]
        g.add_edge_list(edge_list)
    edge_weights = sum(edge_weights, [])
    g.edge_properties["e_weights"] = g.new_edge_property("double", vals=edge_weights)
    return g


def add_node_properties(g, alltrials_categorical_df):
    used_nct_ids = [alltrials_categorical_df.index[i] for i in g.vertex_index]
    alltrials_categorical_df = alltrials_categorical_df.loc[used_nct_ids,:]
    # %%
    for col in tqdm(alltrials_categorical_df.columns):
        categorical_prop = g.new_vertex_property('string')  # Change 'string' to the appropriate data type
        for v in g.vertices():
            categorical_prop[v] = alltrials_categorical_df.loc[list(alltrials_categorical_df.index)[g.vertex_index[v]],col]
            
        g.vertex_properties["v_"+col] = categorical_prop
    return g


def add_node_text_properties(g, alltrials_text_df):
    used_nct_ids = [alltrials_text_df.index[i] for i in g.vertex_index]
    alltrials_text_df = alltrials_text_df.loc[used_nct_ids,:]
    # %%
    for col in tqdm(alltrials_text_df.columns):
        text_prop = g.new_vertex_property('string')  # Change 'string' to the appropriate data type
        for v in g.vertices():
            text_prop[v] = alltrials_text_df.loc[list(alltrials_text_df.index)[g.vertex_index[v]],col]
            
        g.vertex_properties["v_"+col] = text_prop
    return g
