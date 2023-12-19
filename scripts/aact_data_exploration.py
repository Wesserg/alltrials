# %% [markdown]
# # Exploring the aact Cinical Trials dataset
# %% 
# ## Importing libraries
import pandas as pd 
from datetime import datetime as dt
from tqdm import tqdm
import numpy as np
import graph_tool.all as gt
import torch_geometric
from alltrials.etl_utils import get_aact_all_data, get_aact_ct_data, get_aact_selected_data
from alltrials.text_utils import get_text_embeddings
from alltrials.graph_utils import create_ct_graph, add_node_properties, add_node_text_properties
from alltrials.gnn_utils import train_node_classifier, print_results, gt_to_pytorch_geometric
n_samples = 2048
make_figures = True
center_scaling = True


# %%
# Get a list of tables in the ctgov schema
aact_df, col_dict = get_aact_ct_data(aact_table="outcomes", n_rows_limit=n_samples)
# %%

# all_tables_df, col_dict = get_aact_all_data()
# %%
# add list of tables to select [calculated_values, conditions, brief summaries, design_outcomes, designs, detailed_descriptions, eligibilities, outcomes]

selected_tables = ["conditions", "brief_summaries", "calculated_values", "eligibilities", "participant_flows", "designs", "detailed_descriptions"]
selected_tables_df, col_dict = get_aact_selected_data(aact_tables=selected_tables, n_rows_limit=n_samples)

# %%
print(selected_tables_df)
print(col_dict)
# %%
alltrials_df = selected_tables_df
alltrials_text_df = alltrials_df[col_dict['text_columns']]
model_vectors_centerscaled = get_text_embeddings(alltrials_text_df, center_scaling=center_scaling)

# %%
# Add property map to the graph
alltrials_categorical_df = alltrials_df[col_dict['categorical_columns'] + col_dict['numerical_columns']]
g = create_ct_graph(model_vectors_centerscaled, cosine_threshold=0.3)
g = add_node_properties(g, alltrials_categorical_df)
g_text = add_node_text_properties(g, alltrials_text_df)
print(g)

#save graph in gt format
g.save(f"aact_graph_{str(n_samples)}.gt")
# %%
# Draw the graph using graph-tool
if make_figures:
    gt.graph_draw(g, output_size=(1024, 1024), output=f"raw_graph_{str(n_samples)}.png")
    state = gt.minimize_blockmodel_dl(g_text)
    state.draw(output=f"CT_SBM_{str(n_samples)}.png", output_size=(1024, 1024))

    state_nested = gt.minimize_nested_blockmodel_dl(g_text)
    state_nested.draw(
        output=f"CT_nestedSBM_{str(n_samples)}.png",
        vertex_text=g_text.vp["v_conditions_conditions"],
        vertex_font_size=6,
        vertex_text_position="centered",
        output_size=(3072, 3072))


# %%
# %% 
# ## Importing libraries
#load graph from gt format
g = gt.load_graph(f"aact_graph_{str(n_samples)}.gt")
aact_graph_dataset = gt_to_pytorch_geometric(g, alltrials_categorical_df, target_variable="overall_status")
print(aact_graph_dataset[0])


# %% 
DATASET_PATH = "../data"
CHECKPOINT_PATH = "../saved_models/"
cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora")

cora_dataset[0]
# %%

N_LAYERS = 2
DP_RATE = 0.1
# %%
node_mlp_model, node_mlp_result = train_node_classifier(model_name="MLP",
                                                        dataset=aact_graph_dataset,
                                                        c_hidden=16,
                                                        num_layers=N_LAYERS,
                                                        dp_rate=DP_RATE)

print_results(node_mlp_result)
# %%
node_gnn_model, node_gnn_result = train_node_classifier(model_name="GNN_Gconv",
                                                        layer_name="GraphConv",
                                                        dataset=aact_graph_dataset, 
                                                        c_hidden=16, 
                                                        num_layers=N_LAYERS,
                                                        dp_rate=DP_RATE)
print_results(node_gnn_result)
# %%
node_gnn_model, node_gnn_result = train_node_classifier(model_name="GNN_GCN",
                                                        layer_name="GCN",
                                                        dataset=aact_graph_dataset, 
                                                        c_hidden=16, 
                                                        num_layers=N_LAYERS,
                                                        dp_rate=DP_RATE)
print_results(node_gnn_result)

# %%
node_gnn_model, node_gnn_result = train_node_classifier(model_name="GNN_GAT",
                                                        layer_name="GAT",
                                                        dataset=aact_graph_dataset, 
                                                        c_hidden=16, 
                                                        num_layers=N_LAYERS,
                                                        dp_rate=DP_RATE)
print_results(node_gnn_result)
# %%
#load tensorboard
%load_ext tensorboard
# %%
