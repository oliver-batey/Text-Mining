import pandas as pd
import numpy as np
import textacy
import networkx as nx
import matplotlib.pyplot as plt
from itertools import count

with open('news_article.txt', 'r') as file:
    data = file.read().replace('\n', '')   
article = data.replace(u'\xa0', u' ')



#turn sentence into a Doc object 
sent = 'Stocks Hit Record as Biden Calls for More Stimulus'

doc = textacy.make_spacy_doc(sent, lang='en_core_web_sm')
for t in doc:
    print(t.text,t.pos_)

#build list of nodes with number and attributes
nodes = []
for token in doc:
    nodes.append(
                    (token.i,{'text':token.text,
                            'idx':token.i,
                            'pos':token.pos_,
                            'tag':token.tag_,
                            'dep':token.dep_}
                    )
                )

#construct edges from tokens to children
edges = []
for token in doc:
    for child in token.children:
        edges.append((token.i,child.i))

#add nodes and edges to graph
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

#get nodes which are nominal subjects and nodes which are nouns or proper nouns
subjects = [(x,y) for x,y in G.nodes(data=True) if y['dep']=='nsubj']
noun_nodes = [(x,y) for x,y in G.nodes(data=True) if y['pos']=='NOUN' or y['pos']=='PROPN'][1:]

#calculate shortest distance between nominal subjects and nouns
'''
dependencies = []
for (subj,node1) in subjects:
    for (noun,node2) in noun_nodes:
        d = nx.shortest_path(G, source=subj, target=noun)

        dictionary = {'subject':None,'noun':None,'distance':None,'path':None}
        dictionary['subject']=node1['text']
        dictionary['noun']=node2['text']
        dictionary['distance']=len(d)
        dictionary['path']=d
        dependencies.append(dictionary)

#save results in a dataframe
data=pd.DataFrame(dependencies)
data.to_csv('dependency_data_full_doc.csv')
print(data)
'''

bc= nx.betweenness_centrality(G)
node_sizes = 1e5*np.array(list(bc.values()))+5

# create number for each group to allow use of colormap
# get unique groups
groups = set(nx.get_node_attributes(G,'pos').values())
mapping = dict(zip(sorted(groups),count()))
nodes = G.nodes()
colors = [mapping[G.nodes[n]['pos']] for n in nodes]
print(mapping)



#plot and save the graph
fig = plt.figure(figsize=(12,6))
labels_txt = nx.get_node_attributes(G,'text')
labels_pos = nx.get_node_attributes(G,'pos')

keys = []
for key, tag in labels_pos.items():
    if tag == 'NOUN':
        keys.append(key)
labels = {key:labels_txt[key] for key in keys}


pos = nx.spring_layout(G)
#pos = nx.kamada_kawai_layout(G)
ec = nx.draw_networkx_edges(G, pos, edge_color='lightgrey')
nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, 
                            node_size=250, cmap=plt.cm.Pastel2,alpha=0.8)
labels = nx.draw_networkx_labels(G,pos,labels_txt,font_size=12,font_color='dimgrey')
plt.savefig('dependency_network_Sent.png',dpi=300)
plt.savefig('dependency_network_Sent.svg')
#plt.legend(groups,scatterpoints=1)
plt.show()



exit()
