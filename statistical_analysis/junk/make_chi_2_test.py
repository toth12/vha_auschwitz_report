import constants
import pandas as pd
import pdb
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from random import randint
from matplotlib.lines import Line2D
from plotly import express as px
import sys
import scipy.stats as stats
import plotly
import plotly.figure_factory as ff
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.sandbox.stats.multicomp import multipletests





def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def chi2test(df,df_biodata,category):

    df.loc[(df.KeywordLabel == 'food sharing','KeywordID')] = 4289
    df.loc[(df.KeywordLabel == 'food sharing','KeywordLabel')]="camp food sharing"

    df.loc[(df.KeywordLabel == 'friendships','KeywordID')]=14276
    df.loc[(df.KeywordLabel == 'friendships','KeywordLabel')]="friends"
    pdb.set_trace()
    df_category = df_biodata[[category,'IntCode']]
    df = df.merge(df_category,how='left',on="IntCode")
    # Get only categories interested
    

    #df = df.groupby('Gender')["KeywordLabel"].unique().to_frame(name="KeywordLabel").reset_index()
    
    #ff = df.groupby('Gender')["KeywordLabel"].unique().to_frame(name="KeywordLabel").reset_index()
    #cccc = pd.get_dummies(ff.KeywordLabel.apply(pd.Series).stack()).sum(level=0)
    
   
    df = pd.concat([df, df[category].str.get_dummies()], axis=1)

    agg_pipeline = {}
    for element in df[category].unique():
        if pd.isna(element):
            continue
        agg_pipeline[element]='sum'

    
    contingency = df.groupby(['KeywordID','KeywordLabel','IntCode']).agg(agg_pipeline).reset_index()
    pdb.set_trace()
    '''
    for key in agg_pipeline:

        contingency[key] = contingency[key].apply(lambda x: 0 if x <1 else 1)
    '''

    
    contingency = contingency.groupby(['KeywordID',"KeywordLabel"]).agg(agg_pipeline).reset_index()


    total = {}
    for key in agg_pipeline:
        total[key] = len(df_biodata[df_biodata[category]==key])


    #set up the visualization for the markers 
    #contingency = contingency[contingency['KeywordLabel']=="camp food sharing"]
    result = []
    for element in contingency.iterrows():
        total_obs = []
        for key in agg_pipeline:
            number_of_obs = element[1][key]
            number_of_not_obs = total[key] - number_of_obs
            obs = np.array([[number_of_obs,number_of_not_obs]])
            total_obs.append(obs)
        total_obs = np.vstack(total_obs)

        if (~total_obs.any(axis=0)).any():
            continue
        if total_obs.min() <0:
            continue
        

        test_result = chi2_contingency(total_obs)
        
        test_stat = test_result[0]
        p_value = test_result[1]
        

        #Visualize it
        
        plt.figure(figsize=(9, 9))
        df_vis = pd.DataFrame(columns=['expected','observed','county'])
        category_total_counts =  df_biodata.groupby(category)[category].count().to_frame('Count').reset_index()['Count'].array
        df_vis['expected'] = test_result[3][:,0]
        df_vis['observed'] = total_obs[:,0]
        df_vis[category] = agg_pipeline.keys()
        x = np.linspace(0,max(df_vis['expected'].max(),df_vis['observed'].max())+10,10)
        y=x
        plt.plot(x, y)
        ax = sns.scatterplot(y="observed", x="expected", hue=category,data=df_vis, style=category,palette = sns.color_palette(n_colors=len(df_vis)), markers =Line2D.filled_markers[0:len(df_vis)],s=100)
        #ax.text(df_vis['expected'], df_vis['observed'], df_vis['country'], ha='right')
        #
        
        
        plt.ylim(0, max(df_vis['expected'].max(),df_vis['observed'].max())+10)
        plt.xlim(0, max(df_vis['expected'].max(),df_vis['observed'].max())+10)
        
        filename = "_".join(element[1]['KeywordLabel'].split(' '))+'.png'
        plt.legend(loc=4, borderaxespad=0.)
        

        try:
            if category == "Gender":
                plt.savefig(output_directory+'/plots/Gender/'+filename)
            else:
                plt.savefig(output_directory+'/plots/CountryOfBirth/'+filename)
        except:
            pass
        plt.close('all')

        # Calculate expected observed ratio
            
        results_for_individual_cat = total_obs[:,0] / test_result[3][:,0]


        # Calculate odds ratio

        odds_ratios = []
        for i in range(0,len(list(agg_pipeline.keys()))):
            others_present=np.delete(total_obs[:,0],i)
            others_not_present=np.delete(total_obs[:,1],i)
            others = np.array([others_present.sum(),others_not_present.sum()])
            conting = np.concatenate([[total_obs[i]],[others]]).T
            oddsratio, pvalue = stats.fisher_exact(conting)
            odds_ratios.append(oddsratio)



        #results_for_individual_cat = test_result[3][:,0]  / total_obs.sum()

        partial_result = [element[1]['KeywordID'],element[1]['KeywordLabel'],test_stat,p_value]
        partial_result.extend(results_for_individual_cat.tolist())
        partial_result.extend(odds_ratios)
        partial_result.extend(total_obs[:,0].tolist())
        partial_result.extend(test_result[3][:,0].tolist())

        result.append(partial_result)
        if element[1]['KeywordLabel']=="camp food sharing":
            print (p_value)
            
        

    column_labels_observed_expected_ratio = [element+'_observed_expected_ratio' for element in agg_pipeline]
    column_labels_assoc_strength = [element+'_assoc_strength' for element in agg_pipeline]
    column_labels_count_observed = [element+'_count_observed' for element in agg_pipeline]
    column_labels_count_expected = [element+'_count_expected' for element in agg_pipeline]
    columns = ['KeywordID','KeywordLabel','test_stat','p_value']
    columns.extend(column_labels_observed_expected_ratio)
    columns.extend(column_labels_assoc_strength)
    columns.extend(column_labels_count_observed)
    columns.extend(column_labels_count_expected)
    df_chi = pd.DataFrame(result,columns=columns)
    df_chi = df_chi.sort_values('test_stat',ascending=False)
    #df_chi['p_value'] = df_chi['p_value']*len(df_chi)
    df_chi['significance'] = df_chi['p_value']<0.05
    

    #visualize country keyword

    selected_categories = df_chi[df_chi.columns[df_chi.columns.to_series().str.contains('count')]]

    for element in agg_pipeline:
        # select the column about the county
        observed = element+'_count_observed'
        expected= element+'_count_expected'
        ratio = element+'_observed_expected_ratio'
        assoc_strength =  element+'_assoc_strength'
        individual_category = pd.DataFrame()
        individual_category['observed'] = df_chi[observed]
        individual_category['expected'] = df_chi[expected]
        individual_category['KeywordLabel'] = df_chi['KeywordLabel']
        individual_category['ratio'] = df_chi[ratio]
        individual_category['strength'] = df_chi[assoc_strength]


        



        config = dict({'scrollZoom': True})

        plt.ylim(0, max(individual_category['expected'].max(),individual_category['observed'].max())+10)
        plt.xlim(0, max(individual_category['expected'].max(),individual_category['observed'].max())+10)


        fig = px.scatter(individual_category, y="observed",x="expected", hover_data=["KeywordLabel","ratio","strength"],width=800, height=800)
        

        fig.update_xaxes(range=[0, max(individual_category['expected'].max(),individual_category['observed'].max())+100])
        fig.update_yaxes(range=[0, max(individual_category['expected'].max(),individual_category['observed'].max())+100])

       
    
        line=dict(type="line",x0=1,y0=0,x1=max(individual_category['expected'].max(),individual_category['observed'].max())+100,y1=max(individual_category['expected'].max(),individual_category['observed'].max())+100)
        fig.update_layout(shapes=[line])
        xaxis=dict(showspikes = True,spikesnap = 'cursor',showline=True,showgrid=True,spikemode  = 'across',spikedash = 'solid')    
        fig.update_layout(xaxis=xaxis)
        yaxis=dict(showspikes = True,spikesnap = 'cursor',showline=True,showgrid=True,spikemode  = 'across',spikedash = 'solid')    
        fig.update_layout(yaxis=yaxis)


        filename = "_".join(element.split(' '))+'.html'
        plotly.offline.plot(fig, filename=output_directory+'plots/'+filename,config=config,auto_open=False)

    pdb.set_trace()
    df_chi['corrected'] =  multipletests(df_chi['p_value'], method='bonferroni')[0]
    print (df_chi)
    return (df_chi)


if __name__ == '__main__':

    # Read the data



    categories = sys.argv


    if ("Gender" not in categories) and ("CountryOfBirth" not in categories):
        print (categories[1]+ " is not in the categories (Gender or CountryOfBirth) accepted")
        sys.exit()

    category = categories[1]
    input_directory = constants.input_data
    input_file = constants.input_segments_with_simplified_keywords
    bio_data = constants.input_files_biodata_birkenau
    output_directory = constants.output_data_statistical_analysis

    # Read the input data
    df = pd.read_csv(input_directory + input_file)

    # Eliminate those index terms that occur in less than 100 interviews
    kws = df.groupby(['KeywordID', 'KeywordLabel'])['IntCode'].unique().map(lambda x: len(x)).to_frame(name="TotalNumberIntervieweeUsing").reset_index()
    kws_needed = kws[kws.TotalNumberIntervieweeUsing > 0][['KeywordID' , 'KeywordLabel']]

    keywords = kws_needed.reset_index()[['KeywordID', 'KeywordLabel']]
    df = df[df['KeywordID'].isin(kws_needed['KeywordID'])]


    # Filter out Birkenau survivors

    # Get the bio data

    
    bio_data = constants.input_files_biodata_birkenau
    df_biodata = pd.read_csv(input_directory + bio_data)
    

    


    if category == "CountryOfBirth":

        #Filter less frequent country of origins
        count_country = df_biodata.groupby('CountryOfBirth').count()['IntCode'].to_frame(name="Count").reset_index()
        country_to_leave = count_country[count_country['Count']>50]['CountryOfBirth'].to_list()
        df_biodata = df_biodata[df_biodata['CountryOfBirth'].isin(country_to_leave)]


    # Get the IntCode of Jewish survivors
    IntCode = df_biodata.IntCode.to_list()
    IntCode = [str(el) for el in IntCode]


    # Leave only Birkenau survivors
    df = df[df['IntCode'].isin(IntCode)]
    df["IntCode"] = df.IntCode.map(lambda x: int(x))



    result = chi2test(df,df_biodata,category)
    if category =="CountryOfBirth":

        result.to_csv(output_directory+'chi_test_filtered_country_of_birth_with_strenght_of_assoc.csv')
    else:
        result.to_csv(output_directory+'chi_test_filtered_gender_with_strenght_of_assoc.csv')
    


