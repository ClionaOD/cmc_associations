library("vegan")

#set params
n_images <- 1916

#load earlyvis RDM
brain_rdm_path <- list.files(path="/data/movie-associations/brain_rdms/BOLD5000/imgnet_1916", pattern="*.csv", full.names=TRUE, recursive=FALSE)

#load LCH model for 1916 imgnet categories
lch <- read.csv("/data/movie-associations/semantic_measures/imgnet_BOLD5000_lch.csv")
lch <- lch[2:(n_images+1)]
print(dim(lch))

#construct df to store results
# set empty vectors
call <- c()
pearson  <- c()
significance  <- c()

part_call <- c()
part_pearson  <- c()
part_significance  <- c()

for (rdm_file in brain_rdm_path) {
    #set which roi
    roi <- sapply(strsplit(rdm_file,'/'), `[`, 7)
    roi <- sapply(strsplit(roi,'_'), `[`, 1)
    print(sprintf("... %s",roi))
    
    roi_rdm <- read.csv(rdm_file, header=FALSE)
    print(dim(roi_rdm))

    #set which hemisphere to control with
    if (grepl("LH",roi,fixed=TRUE)) {
        earlyvis <- read.csv("/data/movie-associations/brain_rdms/BOLD5000/imgnet_1916/LHEarlyVis_rdm_across_subj.csv", header=FALSE)
        print("... LHEarlyVis being used for control")
        print(dim(earlyvis))
    } else {
        earlyvis <- read.csv("/data/movie-associations/brain_rdms/BOLD5000/imgnet_1916/RHEarlyVis_rdm_across_subj.csv", header=FALSE)
        print("... RHEarlyVis being used for control")
    }
    
    #correlate using mantel test
    mantel <- mantel(roi_rdm, lch)
    #store
    call<-append(call,model)
    pearson<-append(pearson,mantel[3]$statistic)
    significance<-append(significance,mantel[4]$signif)

    #correlate using partial mantel test
    partial <- mantel.partial(roi_rdm,lch,earlyvis)
    #store
    part_call<-append(part_call,model)
    part_pearson<-append(part_pearson,partial[3]$statistic)
    part_significance<-append(part_significance,partial[4]$signif)
}

#save result   
df <- data.frame(Model = call,
            Pearson = pearson,
            Sig = significance
            )
partial_df <- data.frame(Model = part_call,
            Pearson = part_pearson,
            Sig = part_significance
            )
out_df = rbind(df,partial_df)
save_path = sprintf("/data/movie_associations/imgnet_brain_mantel.csv")
write.csv(out_df,sprintf(save_path), row.names = FALSE)