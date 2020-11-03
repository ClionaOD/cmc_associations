# cmc_associations

This is the repository used for SemanticCMC analysis.

get_imagenet.py - for downloading imagenet using their API. Note that there are numerous faulty links, which should be saved to poor_links.txt in order to prevent retries.

get_activations.py - script to get the network activations in response to a set of classes. Activations should be saved in the appropriate location and paths defined (files are too big for the github)

get_lch.py - computes the LCH similarity for the imagenet classes tested (stored in imagenet_categs_X.json or .pickle). Results save to X_categs_lch.pickle.

get_rdms.py - computes the RDMs from the activations returned by get_activations.py

lch_comparison.py - uses scikit bio's Mantel test to correlate RDMs to LCH. This is a full Mantel test.

plot_[].py - various plotting scripts for the analyses.
plot_r_results.py - manually entered results from the R Partial Mantel test, then plot in a bar chart.

./activations - where the network activations from get_activations.py are stored.
./mds - results from MDS (both dataframes and plots)
./rdms - rdm results (pickled dataframes), also figures in ./figs

./results - includes results from lch_comparison (both dataframes and pdf figures)
    results/for_r - activation RDMs as csv files to be read in by R for Partial Mantel

./test_imagenet - a sample of images used to debug the programs.

R_code.txt - summary of the commands used in R to run the Partial Mantel test. 
R_results.txt - tables of results from R mantel tests. Each new table is separated by a '#'header line, split by these to get separate tables showing Pearson correlation and significance. 