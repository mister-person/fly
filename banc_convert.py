import numpy as np
import pandas as pd

n = pd.read_csv("./data/banc_neurons.csv").to_numpy()

n_out = np.concatenate((np.expand_dims(n[:, 0], 1), np.expand_dims(np.full(n.shape[0], True, dtype="O"), 1)), 1)

n_df = pd.DataFrame(n_out, columns=("", "Completed"))
n_df.to_csv("banc_completeness.csv", index=False)

con = pd.read_csv("./data/banc_connections_princeton.csv").to_numpy()
print(con)

id_index = {}
for i, neuron in enumerate(n):
    id_index[neuron[0]] = i

con_dict = {}

nt_types = set()
def nt_to_exc(nt_type):
    map = {'GABA': -1, 'TYR': 1, np.nan: 0, 'SER': 1, 'GLUT': 1, 'ACH': 1, 'DA': 1, 'OCT': 1, 'HIST': 1}
    return map[nt_type]

index = 0

for data in con:
    if (data[0], data[1]) in con_dict:
        entry = con_dict[(data[0], data[1])]
        entry[2] += data[3]
        exc = entry[3]
        entry[4] += exc * data[3]
    else:
        n1_index = id_index[data[0]]
        n2_index = id_index[data[1]]
        nt_type = n[n1_index][3]
        nt_types.add(nt_type)
        con_dict[(data[0], data[1])] = [n1_index, n2_index, data[3], nt_to_exc(nt_type), nt_to_exc(nt_type) * data[3]]

    index += 1 
    if index % 1000000 == 0:
        print(index)

# print(con_dict)
# print(nt_types)

print(len(con_dict))

size = len(con_dict)
con_filtered_np = np.empty((size, 7), dtype="O")
print(con_filtered_np)
for i, item in enumerate(con_dict):
    neuron_data = con_dict[item]
    con_filtered_np[i] = [item[0], item[1], neuron_data[0], neuron_data[1], neuron_data[2], neuron_data[3], neuron_data[4]]

#Presynaptic_ID     Postsynaptic_ID  Presynaptic_Index  Postsynaptic_Index  Connectivity  Excitatory  Excitatory x Connectivity
con_df = pd.DataFrame(con_filtered_np, columns=[
    "Presynaptic_ID",
    "Postsynaptic_ID",
    "Presynaptic_Index",
    "Postsynaptic_Index",
    "Connectivity",
    "Excitatory",
    "Excitatory x Connectivity"
])

print(con_df)
con_df.to_parquet("banc_connectivity.parquet")
