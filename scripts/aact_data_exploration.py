# %% [markdown]
# # Exploring the Cinical Trials dataset
# %% 
# ## Importing libraries
import pandas as pd 
from datetime import datetime as dt
from tqdm import tqdm
import numpy as np
import graph_tool.all as gt

from alltrials.etl_utils import get_aact_all_data, get_aact_studies_data
from alltrials.text_utils import get_text_embeddings
from alltrials.graph_utils import create_ct_graph, add_node_properties
from alltrials.gnn_utils import train_node_classifier, print_results, gt_to_pytorch_geometric
center_scaling = True
n_samples = 1000

# %%
# Get a list of tables in the ctgov schema
studies_df, col_dict = get_aact_studies_data()
# %%
# all_tables_df, col_dict = get_aact_all_data()

# %%
alltrials_df = studies_df
alltrials_text_df = alltrials_df[col_dict['text_columns']]
model_vectors_centerscaled = get_text_embeddings(alltrials_text_df, center_scaling=center_scaling)

# %%
# %%
# Add property map to the graph
alltrials_categorical_df = alltrials_df[col_dict['categorical_columns']]
g = create_ct_graph(model_vectors_centerscaled, cosine_threshold=0.5)
g = add_node_properties(g, alltrials_categorical_df)
print(g)
make_figures = False
# Draw the graph using graph-tool
if make_figures:
    gt.graph_draw(g, output_size=(1024, 1024), output="raw_graph.png")

    state = gt.minimize_blockmodel_dl(g)
    state.draw(output="CT_SBM.png", output_size=(1024, 1024))

    state_nested = gt.minimize_nested_blockmodel_dl(g)
    state_nested.draw(output="CT_nestedSBM.png", output_size=(1024, 1024))


# %%
aact_graph_data = gt_to_pytorch_geometric(g, alltrials_categorical_df)
# %%


# %%
node_mlp_model, node_mlp_result = train_node_classifier(model_name="MLP",
                                                        dataset=aact_graph_data,
                                                        c_hidden=16,
                                                        num_layers=2,
                                                        dp_rate=0.1)

print_results(node_mlp_result)
# %%
node_gnn_model, node_gnn_result = train_node_classifier(model_name="GNN",
                                                        layer_name="GCN",
                                                        dataset=aact_graph_data, 
                                                        c_hidden=16, 
                                                        num_layers=2,
                                                        dp_rate=0.1)
print_results(node_gnn_result)
# %%
