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
from tqdm.auto import tqdm


trajectory_replicates = 25

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
    #TODO: I would prefer to use msmtools because the functions there are unit-tested...
    raise NotImplementedError('consider using msmtools or remove this line.')

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


def print_stationary_distributions(mm, topic_labels):
    #Print the stationary distributions of the top states
    results = []
    for i, element in enumerate(mm.pi.argsort()[::-1]):
        #print (i)
        #print (topic_labels[element])
        #print (mm.pi[element])
        #print ('\n')
        results.append({'topic_name':topic_labels[element],'stationary_prob':mm.pi[element]})
    return results


def calculate_mean_passage_time_between_states(mm, topic_labels):
    #Create a matrix that will hold the data
    #topic_labels = {i:topic_labels[j] for i, j in enumerate(mm.active_set)}
    passage_times = np.zeros(shape=(len(topic_labels),len(topic_labels)))
    df_passage_times = pd.DataFrame(passage_times)  
    for f,row in enumerate(topic_labels):

        for l,column in enumerate(topic_labels):
            try:
                #TODO: why no active set?
                #df_passage_times.iloc()[f][l]=msm.tpt(mm,[mm._full2active[f]],[mm._full2active[l]]).mfpt
                df_passage_times.iloc()[f][l] = msm.tpt(mm,[f],[l]).mfpt
            except:
                df_passage_times.iloc()[f][l]=0
    column_names = {v: k for v, k in enumerate(topic_labels)}
    df_passage_times = df_passage_times.rename(columns=column_names,index=column_names)
    return df_passage_times

def print_mean_passage_time(mm, topic_labels, source,limit = 10):
    
    source_index = topic_labels.index(source)
    #topic_labels_active_set = {i:topic_labels[j] for i, j in enumerate(mm.active_set)}
    
    df_passage_times = pd.DataFrame(topic_labels,columns=['topic_labels']).set_index('index')
    print ('aa')
    mean_ps = []
    for key,label in enumerate(topic_labels):
        try:
            assert mm._full2active[source_index] != -1 and mm._full2active[key] != -1
            mfpt = pyemma.msm.tpt(mm,[mm._full2active[key]],[mm._full2active[source_index]]).mfpt
            mean_ps.append(mfpt)

        except:
            
            mean_ps.append(np.nan)
    df_passage_times['mfpt'] =mean_ps
    df_passage_times = df_passage_times.sort_values('mfpt',ascending=True)
    print ('helloka')
    for i,row in enumerate(df_passage_times[0:limit].iterrows()):
        print (i)
        print (row[1]['topic_labels'])
        print (row[1]['mfpt'])
    



def calculate_flux(mm,topic_labels,source,target,fraction=0.3,print=False):
    #A=[8],B=[2,13],
    # Calculate the flux between two states camp arrival and camp liquidiation / camp transfer )
    np.set_printoptions(suppress=True) 
    A = []
    B = []
    for element in source:
        A.append(topic_labels.index (element))

    for element in target:
         B.append(topic_labels.index(element))
    topic_labels = {i:topic_labels[j] for i, j in enumerate(mm.active_set)}

    assert (-1 not in mm._full2active[A]), 'source states not in active set'
    assert (-1 not in mm._full2active[B]), 'target states not in active set'

    tpt = pyemma.msm.tpt(mm, mm._full2active[A], mm._full2active[B])

    (bestpaths,bestpathfluxes) = tpt.pathways(fraction=fraction)
    cumflux = 0

    # Print the best path between the two states


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
   
    if print:
        for tr in topic_sequences:
            print (tr)
            print (topic_sequences[tr])
    return topic_sequences


    
  

def create_dataframe_with_paths(paths_w,paths_m,filter_stat=None):
    women_topic_sequences = paths_w
    men_topic_sequences = paths_m

    #TODO: should it be ... + list(set(list(men_to...keys())))?
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
        #TODO: should this be ...contains(filter)?
        df = df[df.path.str.contains("social|aid")]
        return df

def estimate_fuzzy_trajectories(step_state_matrix, n_realizations=trajectory_replicates):
    '''
    Convert trajectories using the following approach:
    for each segment with multiple topics, randomly select one topic
    do this for n_realization times and build the MSM on everything
    the error that comes with each random pick should average out this way
    
    returns a trajectory with unassigned -1 states (handled by pyemma)
    '''
    input_data_set = step_state_matrix

    trajs = []
    for _ in range(n_realizations):
        for t in input_data_set:
            _t = (t.cumsum(1) > (np.random.rand(t.shape[0]) * t.sum(1))[:, None]).argmax(1)
            _t[t.sum(1) == 0] = -1
            trajs.append(_t)

    return trajs

def visualize_implied_time_scale(trajectories,output_file):
    its = pyemma.msm.timescales_msm(trajectories, lags=[1, 2, 3, 4, 5, 7, 10], reversible=False,
                                    core_set=np.sort(np.unique(np.concatenate(trajectories)))[1:])
    ax = pyemma.plots.plot_implied_timescales(its, marker='.', xlog=False)
    ax.set_title(output_file.split('/')[-2])
    plt.savefig(output_file)
    plt.close()

def dtraj_assign_to_last_visited_core(trajectories):
    core_set = np.sort(np.unique(np.concatenate(trajectories)))[1:]
    from pyemma.util.discrete_trajectories import rewrite_dtrajs_to_core_sets
    _dtrajs, _, _ = rewrite_dtrajs_to_core_sets(trajectories, core_set=core_set, in_place=False)

    for d in _dtrajs:
        assert d[0] != -1
        while -1 in d:
            mask = (d == -1)
            d[mask] = d[np.roll(mask, -1)]

    assert -1 not in np.unique(np.concatenate(_dtrajs))

    return _dtrajs

def visualize_implied_time_scale_bayes(trajectories, output_file, n_realizations=trajectory_replicates):
    """
    THIS IS EXPERIMENTAL!
    Estimating errors from core set MSM with multiple realizations per observed
    trajectory as we did it here is not an established method. Use with caution and
    only for error estimate.
    """

    _dtrajs = dtraj_assign_to_last_visited_core(trajectories)

    # error estimation needs decorallated samples, i.e.
    # cannot use multiple copies per interview. take a random one instead.
    # this is experimental
    its = pyemma.msm.timescales_msm(_dtrajs[::n_realizations],
                                    lags=[1, 2, 3, 4, 5, 7, 10], reversible=False,
                                    errors='bayes', nsamples=10)
    ax = pyemma.plots.plot_implied_timescales(its, marker='.', xlog=False)
    ax.set_title(output_file.split('/')[-2])
    plt.savefig(output_file)
    plt.close()


def estimate_markov_model_from_trajectories(trajectories, msmlag=2):

    msm = pyemma.msm.estimate_markov_model(trajectories, msmlag, reversible=False,
                                           core_set=np.sort(np.unique(np.concatenate(trajectories)))[1:])
    return msm

def estimate_bayesian_markov_model_from_trajectories(trajectories, msmlag=2, n_realizations=trajectory_replicates):
    """
    THIS IS EXPERIMENTAL!
    Estimating errors from core set MSM with multiple realizations per observed
    trajectory as we did it here is not an established method. Use with caution and
    only for error estimate.
    """
    _dtrajs = dtraj_assign_to_last_visited_core(trajectories)
    msm = pyemma.msm.bayesian_markov_model(_dtrajs[::n_realizations], msmlag, reversible=False)
    return msm

def prepare_histogram_to_compare_stationary_distribution_with_plausi_measure(msm, output_file):
    # histogram to compare stationary distribution with (plausibility measure)
    hist = np.bincount(np.concatenate(msm.dtrajs_full))
    plt.plot(msm.pi)
    plt.plot(hist[msm.active_set]/ hist[msm.active_set].sum(), alpha=.5)
    plt.savefig(output_file)
    plt.close()

def visualize_msm_graph(msm,features_df,output_file,
                        tmat_thresh=2e-2):

    _m = np.zeros_like(msm.transition_matrix)

    _m[msm.transition_matrix > tmat_thresh] = msm.transition_matrix[msm.transition_matrix > tmat_thresh]
    g_tmat = nx.from_numpy_array(_m, create_using=nx.DiGraph)
    nodename_dict = {i:features_df.iloc[j].KeywordLabel for i, j in enumerate(msm.active_set)}

    g_tmat = nx.relabel_nodes(g_tmat, nodename_dict)

    edge_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("uwe",[(0, 0, 0, .1), (0, 0, 0, 1)])

    weights = np.array(list(nx.get_edge_attributes(g_tmat, 'weight').values()))
    pos = nx.fruchterman_reingold_layout(g_tmat, k=1e-1)# fixed=keep)
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(g_tmat, pos, node_size=msm.pi*1000, ax=ax, )
    nx.draw_networkx_edges(g_tmat, pos, edge_cmap=edge_cmap, node_size=msm.pi*1000,
                        edge_color=weights, width=2, ax=ax);

    fig.savefig(output_file)

def get_tpt_major_flux(msm,features_df,KeywordLabel_A,KeywordLabel_B,fraction=.9):
    if not isinstance(KeywordLabel_A, list):
        KeywordLabel_A = [KeywordLabel_A]
    if not isinstance(KeywordLabel_B, list):
        KeywordLabel_B = [KeywordLabel_B]

    A = features_df[features_df['KeywordLabel'].isin(KeywordLabel_A)].index.to_numpy()
    B = features_df[features_df['KeywordLabel'].isin(KeywordLabel_B)].index.to_numpy()

    assert -1 not in msm._full2active[A] and -1 not in msm._full2active[B]

    tpt = pyemma.msm.tpt(msm, msm._full2active[A], msm._full2active[B])
    fl = tpt.major_flux(fraction)

    # Draw the graph
    g = nx.from_numpy_array(fl, create_using=nx.DiGraph)
    nodename_dict = {i:features_df.iloc[j].KeywordLabel for i, j in enumerate(msm.active_set)}
    g = nx.relabel_nodes(g, nodename_dict)
    
    return g, tpt, nodename_dict

def visualize_tpt_major_flux(msm,features_df,KeywordLabel_A,KeywordLabel_B,output_file,fraction=.9):

    g, _, nodename_dict = get_tpt_major_flux(msm,features_df,KeywordLabel_A,KeywordLabel_B,fraction=fraction)

    edge_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("uwe",[(0, 0, 0, .1), (0, 0, 0, 1)])

    labelthres = .01
    labels = {}

    for ind, (node, prob) in enumerate(zip(g.nodes, msm.pi)):
        if prob > labelthres:# or ind in most_important_nodes:
            labels[node] = node.replace(' ', '\n')
            
        elif node in features_df.iloc[A].KeywordLabel.to_list() + features_df.iloc[B].KeywordLabel.to_list():
            labels[node] = node.replace(' ', '\n').upper()
            
        else:
            labels[node] = ''


    _c = msmtools.analysis.committor(msm.transition_matrix, msm._full2active[A], msm._full2active[B])

    init_pos = np.random.rand(g.number_of_nodes(), 2)
    init_pos[:, 0] = _c


    init_pos_dict = {node:pos for pos, node in zip(init_pos, g.nodes)}
    weights = np.array(list(nx.get_edge_attributes(g, 'weight').values()))
    pos = nx.fruchterman_reingold_layout(g, iterations=10, k=5e-3, pos=init_pos_dict)#) fixed=keep)
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(g, pos, node_size=msm.pi*1000, ax=ax, )
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=14)
    nx.draw_networkx_edges(g, pos, edge_cmap=edge_cmap, node_size=msm.pi*1000,
                        edge_color=weights, width=2, ax=ax);

    if output_file is not None:
        fig.savefig(output_file)

    return g, pos, nodename_dict


def visualize_most_important_paths(msm, fraction, features_df, KeywordLabel_A, KeywordLabel_B, output_directory=None,gender="m"):
    if not isinstance(KeywordLabel_A, list):
        KeywordLabel_A = [KeywordLabel_A]
    if not isinstance(KeywordLabel_B, list):
        KeywordLabel_B = [KeywordLabel_B]

    A = features_df[features_df['KeywordLabel'].isin(KeywordLabel_A)].index.to_numpy()
    B = features_df[features_df['KeywordLabel'].isin(KeywordLabel_B)].index.to_numpy()

    assert -1 not in msm._full2active[A] and -1 not in msm._full2active[B]

    nodename_dict = {i:features_df.iloc[j].KeywordLabel for i, j in enumerate(msm.active_set)}
    tpt = pyemma.msm.tpt(msm, msm._full2active[A], msm._full2active[B])

    paths, capacities = tpt.pathways(fraction=fraction)
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

    pos = nx.fruchterman_reingold_layout(pathgraph,iterations=20, k=1.8e-2, pos=init_pos_dict, )#fixed=keep)


    fig, ax = plt.subplots(figsize=(10, 10))
    if gender == 'm':
        color = '#1f77b4'
    else:
        color = '#ff7f0e'
    nx.draw_networkx_nodes(pathgraph, pos, node_size=msm.pi[pathg_nodes]*10000, ax=ax, node_color=color)
    nx.draw_networkx_labels(pathgraph, pos, labels=labels, font_size=16)

    nx.draw_networkx_edges(pathgraph, pos, node_size=msm.pi[pathg_nodes]*10000,
                           edge_cmap=edge_cmap, 
                        edge_color=weights, width=2, ax=ax)
        
    if output_directory:
        output_file_name = 'most_imp_path_'+KeywordLabel_A+'_'+KeywordLabel_B + '_'+str(fraction)+'.png'
        output = output_directory + '/' + output_file_name
        fig.savefig(output)

    return pathgraph, pos, nodename_dict



def estimate_pi_error(dtrajs, orig_msm, ntrails=10, conf_interval=0.68):
    """
    Estimate boostrap error for stationary probability

    :param dtrajs: list of np.array, discrete trajectories
    :param orig_msm: pyemma.msm.MarkovModel
    Only used for reference of lag time and to incorporate ML
    stationary distribution to data frame
    :param ntrails: int, the number of bootstrap samples to draw.
    :param conf_interval: float 0 < conf_interval < 1

    :return:
    pandas.DataFrame instance containing ML MSM pi and bootstrap error
    """
    from pyemma.util.statistics import confidence_interval

    pi_samples = np.zeros((ntrails, orig_msm.nstates))

    for trial in tqdm(range(ntrails)):
        try:
            bs_sample = np.random.choice(len(dtrajs),
                                         size=len(dtrajs),
                                         replace=True)
            dtraj_sample = list(np.array(dtrajs)[bs_sample])

            msm = pyemma.msm.estimate_markov_model(dtraj_sample,
                                                   lag=orig_msm.lag)

            pi_samples[trial, msm.active_set] = msm.pi
        except Exception as e:
            print(e)

    std = pi_samples.std(axis=0)
    lower_confidence, upper_confidence = confidence_interval(pi_samples, conf_interval)

    probabilities = pd.DataFrame(np.array([orig_msm.active_set,
                                           orig_msm.pi,
                                           std,
                                           lower_confidence,
                                           upper_confidence]).T,
                                 columns=['State',
                                          'StatDist',
                                          'Std',
                                          'LowerConf',
                                          'UpperConf'])

    # type cast to int
    probabilities['State'] = probabilities['State'].astype(int)

    return probabilities


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def split_interview(interview, max_gap=1):
    all_empty_segments = consecutive(np.where(interview.sum(axis=1) == 0)[0])
    empty_segs = [gap for gap in all_empty_segments if len(gap) > max_gap]

    if len(empty_segs) > 0:
        int_indices = np.arange(interview.shape[0])
        int_indices = int_indices[~np.in1d(int_indices, np.concatenate(empty_segs))]
        segment_index_list = consecutive(int_indices)
        return [interview[idx] for idx in segment_index_list]
    else:
        return [interview]


if __name__ == '__main__':
    pass

