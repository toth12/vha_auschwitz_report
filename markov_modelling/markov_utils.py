import pandas as pd 
import pdb
from itertools import islice
import numpy as np
from pyemma import msm
import msmtools
from msmtools.estimation import connected_sets
import networkx as nx
import pyemma
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

def window(seq, n=2):
    "Sliding window width n from seq.  From old itertools recipes."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def cg_transition_matrix(T, chi):
    """
    Map a transition matrix T to coarse states via membership
    matrix chi. Implements Eq. 14 of
    Roeblitz & Weber, Adv Data Anal Classif (2013) 7:147–179
    DOI 10.1007/s11634-013-0134-6
    
    :params:
    T: np.ndarray; transition matrix in microstate space
    chi: np.ndarray membership matrix
    """
    assert msmtools.analysis.is_connected(T)

    pi = msmtools.analysis.stationary_distribution(T)
    D2 = np.diag(pi)
    D_c2_inv = np.diag(1/np.dot(chi.T, pi))
    return D_c2_inv @ chi.T @ D2 @ T @ chi

def transform_transition_matrix_connected(transition_matrix):
    
    connected_nodes = connected_sets(transition_matrix)[0]
    connected_matrix = np.take(transition_matrix,connected_nodes,axis=0)
    connected_matrix = np.take(connected_matrix,connected_nodes,axis=1)

    
    #removed_nodes = [element[0] for element in connected_sets(transition_matrix)[1:]]
    removed_nodes = [element for element in range(0,transition_matrix.shape[0]) if element not in connected_nodes]
    removed_nodes.sort()
    removed_nodes.reverse()
    return connected_matrix,removed_nodes

def round_to_one(m):
    indices=np.where(m.sum(axis=1)>1)
    for element in indices: 
        sums = m[element].sum()
        dif = sums-1
        x = np.nonzero(m[element])[0][0]
        m[element][x]=m[element][x]-dif

    indices=np.where(m.sum(axis=1)<1)
    for element in indices:
        sums = m[element].sum()
        dif = 1-sums
        x = np.nonzero(m[element])[0][0]
        m[element][x]=m[element][x]+dif
    return m


def train_markov_chain(transition_matrix):
    mm = msm.markov_model(transition_matrix)
    #mm = msm.MSM(transition_matrix,neig=13)
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6910584/

    # Use this to check if reversible
    # TODO: not sure if this attribute is updated internally; if so: assert
    mm.is_reversible

    

    return mm

def print_stationary_distributions(mm, topic_labels):
    #Print the stationary distributions of the top states
    results = []
    for i, element in enumerate(mm.pi.argsort()[::-1]):
        #print (i)
        #print (topic_labels[element])
        #print (mm.pi[element])
        #print ('\n')
        results.append({'topic_name':topic_labels[mm.active_set[element]],'stationary_prob':mm.pi[element]})
    return results

    # Print the eigenvalues of states

    '''for element in mm.eigenvalues().argsort()[::-1]:
        print (topic_list[element])
    '''
def calculate_mean_passage_time_between_states(mm,topic_labels):
    #Create a matrix that will hold the data

    passage_times = np.zeros(shape=(len(topic_labels),len(topic_labels)))
    df_passage_times = pd.DataFrame(passage_times)  
    for f,row in enumerate(topic_labels):

        for l,column in enumerate(topic_labels):
            try:
                df_passage_times.iloc()[f][l]=msm.tpt(mm,[f],[l]).mfpt
            except:
                df_passage_times.iloc()[f][l]=0
    column_names = {v: k for v, k in enumerate(topic_labels)}
    df_passage_times = df_passage_times.rename(columns=column_names,index=column_names)
    return df_passage_times

def calculate_flux(mm,topic_labels,source,target):
    #A=[8],B=[2,13],
    # Calculate the flux between two states camp arrival and camp liquidiation / camp transfer )
    np.set_printoptions(suppress=True) 
    A = []
    B = []
    for element in source:
        A.append(topic_labels.index(element))

    for element in target:
         B.append(topic_labels.index(element))
   
    tpt = msm.tpt(mm, A, B)

    nCut = 1
    (bestpaths,bestpathfluxes) = tpt.pathways(fraction=0.3)
    cumflux = 0

    # Print the best path between the two states

    print("Path flux\t\t%path\t%of total\tpath")

    topic_sequences = {}
    for i in range(len(bestpaths)):
        cumflux += bestpathfluxes[i]
        flux = 100.0*bestpathfluxes[i]/tpt.total_flux
        if flux > 0:
            #print(bestpathfluxes[i],'\t','%3.1f'%(100.0*bestpathfluxes[i]/tpt.total_flux),'%\t','%3.1f'%(100.0*cumflux/tpt.total_flux),'%\t\t',bestpaths[i])
            
            topic_sequence = []
            for element in bestpaths[i]:
                #print (topic_labels[element])
                topic_sequence.append(topic_labels[element])
            topic_sequence = '-'.join(topic_sequence)
            topic_sequences[topic_sequence]=100.0*bestpathfluxes[i]/tpt.total_flux
   
    return topic_sequences


def calculate_flux_2(mm,topic_labels,source,target):
    #A=[8],B=[2,13],
    # Calculate the flux between two states camp arrival and camp liquidiation / camp transfer )
    np.set_printoptions(suppress=True) 
    A = []
    B = []
    for element in source:
        A.append(topic_labels.index(element))

    for element in target:
         B.append(topic_labels.index(element))
   
    tpt = pyemma.msm.tpt(mm, mm._full2active[A], mm._full2active[B])

    nCut = 1
    (bestpaths,bestpathfluxes) = tpt.pathways(fraction=0.3)
    cumflux = 0

    # Print the best path between the two states

    print("Path flux\t\t%path\t%of total\tpath")

    topic_sequences = {}
    for i in range(len(bestpaths)):
        cumflux += bestpathfluxes[i]
        flux = 100.0*bestpathfluxes[i]/tpt.total_flux
        if flux > 0:
            #print(bestpathfluxes[i],'\t','%3.1f'%(100.0*bestpathfluxes[i]/tpt.total_flux),'%\t','%3.1f'%(100.0*cumflux/tpt.total_flux),'%\t\t',bestpaths[i])
            
            topic_sequence = []
            for element in bestpaths[i]:
                #print (topic_labels[element])
                topic_sequence.append(topic_labels[element])
            topic_sequence = '-'.join(topic_sequence)
            topic_sequences[topic_sequence]=100.0*bestpathfluxes[i]/tpt.total_flux
   
    return topic_sequences


    
  

def create_dataframe_with_paths(paths_w,paths_m,filter_stat=None):
    women_topic_sequences = paths_w
    men_topic_sequences = paths_m

    pathes = list(set(list(women_topic_sequences.keys())+list(men_topic_sequences.keys())))

    result = []
    for element in pathes:
        try:
            wflux = women_topic_sequences[element]
        except:
            wflux = 0
        try:
            mflux = men_topic_sequences[element]
        except:
            mflux = 0
        result.append({'path':element,'wflux':wflux,'mflux':mflux})
    if filter_stat == None:
        return pd.DataFrame(result)
    else:
        filter = "|".join(filter_stat)
        df = pd.DataFrame(result)
        df = df[df.path.str.contains("social|aid")]
        return df

def post_process_topic_sequences(sequences,features_df):
    final_result = []
    for element in sequences:
        pdb.set_trace()

    return final_result


def estimate_fuzzy_trajectories(step_state_matrix):
    '''
    Convert trajectories using the following approach:
    for each segment with multiple topics, randomly select one topic
    do this for n_realization times and build the MSM on everything
    the error that comes with each random pick should average out this way
    
    this cell also assigns the last visited state if a frame is empty (row.sum() == 0)
    '''
    input_data_set = step_state_matrix
    n_realizations = 50
    trajs = []
    for _ in tqdm(range(n_realizations)):
        for t in input_data_set:

            _t = np.zeros_like(t)
            for n, row in enumerate(t):
                if row.sum() > 1:
                    rindx = np.random.choice(row.shape[0], size=1, p=row/row.sum())
                    _t[n, rindx] = 1
                elif row.sum() == 0:
                    _t[n] = _t[n-1]
                else:
                    _t[n] = row

            # make sure this is binary assignment
            assert np.unique(np.sum(_t, axis=1)) == np.array([1]), 'failed binary assignment'

            # convert one-hot encoding to state definitions
            trajs.append(np.argmax(_t, axis=1) )
    return trajs

def visualize_implied_time_scale(trajectories,output_file):
    its = pyemma.msm.timescales_msm(trajectories, lags=np.arange(1, 50, 5), reversible=False)
    pyemma.plots.plot_implied_timescales(its, marker='.', xlog=False)
    plt.savefig(output_file)
    plt.close()

def estimate_markov_model_from_trajectories(trajectories):
    msm = pyemma.msm.estimate_markov_model(trajectories, 10, reversible=False)
    return msm

def prepare_histogram_to_compare_stationary_distribution_with_plausi_measure(msm,trajectories,output_file):
    # histogram to compare stationary distribution with (plausibility measure)
    hist = np.bincount(np.concatenate(trajectories))
    plt.plot(msm.pi)
    plt.plot(hist[msm.active_set]/ hist.sum(), alpha=.5)
    plt.savefig(output_file)
    plt.close()

def visualize_msm_graph(msm,features_df,KeywordLabel_A,KeywordLabel_B,output_file):
    A = features_df[features_df['KeywordLabel'].isin([KeywordLabel_A])].index.to_numpy()
    B = features_df[features_df['KeywordLabel'] == KeywordLabel_B].index.to_numpy()
    tpt = pyemma.msm.tpt(msm, msm._full2active[A], msm._full2active[B])
    fl = tpt.major_flux()
    g = nx.from_numpy_array(fl, create_using=nx.DiGraph)

    _m = np.zeros_like(msm.transition_matrix)
    tmat_thresh = 2e-2
    _m[msm.transition_matrix > tmat_thresh] = msm.transition_matrix[msm.transition_matrix > tmat_thresh]
    g_tmat = nx.from_numpy_array(_m, create_using=nx.DiGraph)
    nodename_dict = {i:features_df.iloc[j].KeywordLabel for i, j in enumerate(msm.active_set)}
    g = nx.relabel_nodes(g, nodename_dict)
    g_tmat = nx.relabel_nodes(g_tmat, nodename_dict)
    edge_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("uwe",[(0, 0, 0, .1), (0, 0, 0, 1)])
    paths = tpt.pathways(fraction=.66)
    important_nodes = np.unique(np.concatenate(paths[0]))
    node_list = [list(g.nodes())[i] for i in important_nodes]

    labelthres = .01
    labels = {}

    for ind, (node, prob) in enumerate(zip(g.nodes, msm.pi)):
        if prob > labelthres:# or ind in most_important_nodes:
            labels[node] = node.replace(' ', '\n')
            
        elif node in features_df.iloc[A].KeywordLabel.to_list() +         features_df.iloc[B].KeywordLabel.to_list():
            labels[node] = node.replace(' ', '\n').upper()
            
        else:
            labels[node] = ''
    print(f'{sum([l != "" for l in labels.values()])} labels to show')
    weights = np.array(list(nx.get_edge_attributes(g_tmat, 'weight').values()))
    pos = nx.fruchterman_reingold_layout(g_tmat, k=1e-1)# fixed=keep)
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(g_tmat, pos, node_size=msm.pi*1000, ax=ax, )
    nx.draw_networkx_edges(g_tmat, pos, edge_cmap=edge_cmap, node_size=msm.pi*1000,
                        edge_color=weights, width=2, ax=ax);

    fig.savefig(output_file)

def visualize_tpt_major_flux(msm,features_df,KeywordLabel_A,KeywordLabel_B,output_file):
    A = features_df[features_df['KeywordLabel'].isin([KeywordLabel_A])].index.to_numpy()
    B = features_df[features_df['KeywordLabel'] == KeywordLabel_B].index.to_numpy()
    tpt = pyemma.msm.tpt(msm, msm._full2active[A], msm._full2active[B])
    fl = tpt.major_flux()

    # Draw the graph
    g = nx.from_numpy_array(fl, create_using=nx.DiGraph)
    _m = np.zeros_like(msm.transition_matrix)
    tmat_thresh = 2e-2
    _m[msm.transition_matrix > tmat_thresh] = msm.transition_matrix[msm.transition_matrix > tmat_thresh]
    g_tmat = nx.from_numpy_array(_m, create_using=nx.DiGraph)
    nodename_dict = {i:features_df.iloc[j].KeywordLabel for i, j in enumerate(msm.active_set)}
    g = nx.relabel_nodes(g, nodename_dict)
    g_tmat = nx.relabel_nodes(g_tmat, nodename_dict)
    edge_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("uwe",[(0, 0, 0, .1), (0, 0, 0, 1)])
    paths = tpt.pathways(fraction=.66)
    important_nodes = np.unique(np.concatenate(paths[0]))
    node_list = [list(g.nodes())[i] for i in important_nodes]

    labelthres = .01
    labels = {}

    for ind, (node, prob) in enumerate(zip(g.nodes, msm.pi)):
        if prob > labelthres:# or ind in most_important_nodes:
            labels[node] = node.replace(' ', '\n')
            
        elif node in features_df.iloc[A].KeywordLabel.to_list() +         features_df.iloc[B].KeywordLabel.to_list():
            labels[node] = node.replace(' ', '\n').upper()
            
        else:
            labels[node] = ''
    print(f'{sum([l != "" for l in labels.values()])} labels to show')




    _c = msmtools.analysis.committor(msm.transition_matrix, msm._full2active[A], msm._full2active[B])

    init_pos = np.random.rand(g.number_of_nodes(), 2)
    init_pos[:, 0] = _c


    init_pos_dict = {node:pos for pos, node in zip(init_pos, g.nodes)}
    weights = np.array(list(nx.get_edge_attributes(g, 'weight').values()))
    pos = nx.fruchterman_reingold_layout(g, iterations=10, k=5e-3, pos=init_pos_dict)#) fixed=keep)
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(g, pos, node_size=msm.pi*1000, ax=ax, )
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=9)
    nx.draw_networkx_edges(g, pos, edge_cmap=edge_cmap, node_size=msm.pi*1000,
                        edge_color=weights, width=2, ax=ax);

    fig.savefig(output_file)


def visualize_most_important_paths(msm,features_df,KeywordLabel_A,KeywordLabel_B,output_file):
    A = features_df[features_df['KeywordLabel'].isin([KeywordLabel_A])].index.to_numpy()
    B = features_df[features_df['KeywordLabel'] == KeywordLabel_B].index.to_numpy()
    nodename_dict = {i:features_df.iloc[j].KeywordLabel for i, j in enumerate(msm.active_set)}
    tpt = pyemma.msm.tpt(msm, msm._full2active[A], msm._full2active[B])
    paths, capacities = tpt.pathways(fraction=.25)
    pathgraph = nx.DiGraph()
    pathg_node_names = []
    pathg_nodes = []
    for path, cap in zip(paths, capacities):
        for step in range(len(path)-1):
            w = cap
            _w = pathgraph.get_edge_data(path[step], path[step+1])
            if _w is not None:
                w += _w['weight']
            pathgraph.add_edge(nodename_dict[path[step]], 
                               nodename_dict[path[step+1]], weight=w)
            if nodename_dict[path[step]] not in pathg_node_names:
                pathg_node_names.append(nodename_dict[path[step]])
                pathg_nodes.append(path[step])
            if nodename_dict[path[step+1]] not in pathg_node_names:
                pathg_node_names.append(nodename_dict[path[step+1]])
                pathg_nodes.append(path[step+1])
    edge_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("uwe", 
                                                                        [(0, 0, 0, .1), (0, 0, 0, 1)])
    labelthres = .000
    labels = {}

    for idx, node in zip(pathg_nodes, pathg_node_names):

        if node not in pathgraph.nodes():
            continue
        
            
        if node in features_df.iloc[A].KeywordLabel.to_list() + features_df.iloc[B].KeywordLabel.to_list():
            labels[node] = node.replace(' ', '\n').upper()
        elif msm.pi[msm._full2active[idx]] > labelthres:
            labels[node] = node.replace(' ', '\n')
            
        else:
            labels[node] = ''

    print(f'{sum([l != "" for l in labels.values()])} labels to show')
    _c = msmtools.analysis.committor(msm.transition_matrix, msm._full2active[A], msm._full2active[B])
    init_pos = np.random.rand(pathgraph.number_of_nodes(), 2)
    init_pos[:, 0] = _c[pathg_nodes]
    init_pos_dict = {node:[_c[idx], np.random.rand()] for node, idx in zip(pathg_node_names, pathg_nodes)}
    weights = np.array(list(nx.get_edge_attributes(pathgraph, 'weight').values())).astype(float)
    pos = init_pos_dict
    pos = nx.fruchterman_reingold_layout(pathgraph,iterations=20, k=1.8e-2, pos=init_pos_dict, )#fixed=keep)


    fig, ax = plt.subplots(figsize=(10, 10))

    nx.draw_networkx_nodes(pathgraph, pos, node_size=msm.pi[pathg_nodes]*10000, ax=ax, )
    nx.draw_networkx_labels(pathgraph, pos, labels=labels, font_size=9)
    nx.draw_networkx_edges(pathgraph, pos, node_size=msm.pi[pathg_nodes]*10000,
                           edge_cmap=edge_cmap, 
                        edge_color=weights, width=2, ax=ax);

    fig.savefig(output_file)







if __name__ == '__main__':
    pass

