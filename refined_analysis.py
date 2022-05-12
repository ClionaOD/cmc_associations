import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from get_dend import hierarchical_clustering
from scipy.spatial import procrustes
import scipy.spatial.distance as ssd
from sklearn.manifold import MDS
from skbio.stats.distance import mantel

rdm_pth = '/data/movie-associations/rdms/defined_categs'

layers = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']
#layers = ['conv5']

plot_rdm = False
plot_cluster = False
plot_mds = False
mantel_test = True

context_model = pd.read_csv('/data/movie-associations/rdms/defined_categs/context_model.csv',index_col=0)
tripartite_model = pd.read_csv('/data/movie-associations/rdms/defined_categs/tripartite_model.csv',index_col=0)

mantel_context = pd.DataFrame(index=layers)
mantel_tripartite = pd.DataFrame(index=layers)

for layer in layers:
    
    ref = False
    
    for model_rdm in [i for i in os.listdir(f'{rdm_pth}/{layer}') if '.csv' in i]:
        rdm = pd.read_csv(f'{rdm_pth}/{layer}/{model_rdm}', index_col=0)
        
        # normalize
        max = rdm.values.max() ; min = rdm.values.min()
        rdm = (rdm - min) / max - min

        model_name = model_rdm.split("_")[0]
        
        if plot_rdm:
            sns.heatmap(rdm)
            plt.savefig(f'{rdm_pth}/{layer}/{model_name}_rdm.png')
            plt.close()

        if plot_cluster:
            outpath = f'{rdm_pth}/{layer}/{model_name}_euclidean_dendrogram.png'
            hierarchical_clustering(rdm.values, rdm.index, outpath)

        if plot_mds:
            #do MDS on the euclidean distance matrix, set category for plot
            mds = MDS(n_components=2, dissimilarity='precomputed')
            df_embedding = pd.DataFrame(mds.fit_transform(rdm.values), index=rdm.index)
            if ref == False:
                align = df_embedding.values
                ref = True
            else:
                mtx1, mtx2, disparity = procrustes(align, df_embedding.values)
                df_embedding = pd.DataFrame(mtx2, index=rdm.index)
            
            df_embedding['category'] = rdm.index

            fig, ax = plt.subplots(figsize=(12,12))
            sns.scatterplot(
                x=df_embedding[0],
                y=df_embedding[1], 
                hue=df_embedding['category'],
                legend=True,  
                ax=ax)
            ax.set_title(model_name)
            ax.set_xlabel(' ')
            ax.set_ylabel(' ')
            ax.axis('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #plt.show()
            plt.savefig(f'{rdm_pth}/{layer}/{model_name}_mds.png')
            plt.close()
        
        if mantel_test:
            
            coeff_context, p_value_context, _ = mantel(rdm.values, context_model.values)
            mantel_context.loc[layer,model_name] = coeff_context
            
            coeff_tripart, p_value_tripart, _ = mantel(rdm.values, tripartite_model.values)
            mantel_tripartite.loc[layer,model_name] = coeff_tripart

            
            '''print(f"""
            ===========================
            Results for {model_name} layer {layer}:
                context r = {coeff_context}     p = {p_value_context}
                tripartite r = {coeff_tripart}  p = {p_value_tripart}
            """)'''

if mantel_test:
    drop_cols = ['random-distort', 'movieLab', 'random-supervised','authorLab','random-Lab']
    context_plot = mantel_context.drop(drop_cols,axis=1)
    tripartite_plot = mantel_tripartite.drop(drop_cols,axis=1)
    
    fig,((ax1,leg),(ax2,_)) = plt.subplots(nrows=2, ncols=2, figsize=(11.69/1.25,8.27/1.5), sharex=True)
    
    sns.lineplot(data=context_plot.astype(float), ax=ax1, dashes=False)
    sns.lineplot(data=tripartite_plot.astype(float), ax=ax2, dashes=False)
    
    handles, labels = ax1.get_legend_handles_labels()
    ax1.get_legend().remove() ; ax2.get_legend().remove()
    ax1.spines['top'].set_visible(False) ; ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False) ; ax2.spines['right'].set_visible(False)
    ax1.set_xlabel('alexnet layer') ; ax2.set_xlabel('alexnet layer')
    ax1.set_ylabel('Correlation to context model\n(Pearson\'s r)')
    ax2.set_ylabel('Correlation to tripartite model\n(Pearson\'s r)')
    leg.legend(handles, labels)
    
    leg.axis('off')
    _.axis('off') 
    
    plt.savefig('model_correlations.pdf')
    

