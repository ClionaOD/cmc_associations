library("vegan")
 
layers <- c('conv1','conv2','conv3','conv4','conv5','fc6','fc7')
bold_path <- list.files(path="/data/movie-associations/brain_rdms/BOLD5000/imgnet_1916", pattern="*.csv", full.names=TRUE, recursive=FALSE)
test_path <- "/data/movie-associations/rdms/segmentation/obj_trained/imgnet_bold"

n_images = 1916

for (layer in layers) {
    print(sprintf("=============== LAYER %s ===============", layer))
    
    # read lists for model rdms and brain rdms
    pth <- sprintf("%s/%s",test_path,layer)
    rdm_path <- list.files(path=pth, pattern="*.csv", full.names=TRUE, recursive=FALSE)
    
    # read in random for partial mantel
    random <- read.csv(sprintf("%s/random_%s.csv",pth,layer))

    for (roi_file in bold_path){
        # get roi as characters
        filename <- sapply(strsplit(roi_file,'/'), `[`, 7)
        roi <- sapply(strsplit(filename,'_'), `[`, 1)
        print(sprintf("Working on %s region",roi))
        roi_rdm <- read.csv(roi_file, header=FALSE) # brain rdms are np. arrays, don't need to slice

        # set empty vectors
        call <- c()
        pearson  <- c()
        significance  <- c()

        part_call <- c()
        part_pearson  <- c()
        part_significance  <- c()

        for (rdm_file in rdm_path) {
            model <- sapply(strsplit(rdm_file,'/'), `[`, 9)
            model <- sapply(strsplit(model,'_'), `[`, 1)
            print(sprintf("... %s",model))
            
            rdm <- read.csv(rdm_file)
            mantel <- mantel(rdm[2:(n_images+1)], roi_rdm)
            
            call<-append(call,model)
            pearson<-append(pearson,mantel[3]$statistic)
            significance<-append(significance,mantel[4]$signif)
            
            partial <- mantel.partial(rdm[2:(n_images+1)],roi_rdm,random[2:(n_images+1)])
            
            part_call<-append(part_call,model)
            part_pearson<-append(part_pearson,partial[3]$statistic)
            part_significance<-append(part_significance,partial[4]$signif)
        }
        
        df <- data.frame(Model = call,
                    Pearson = pearson,
                    Sig = significance
                    )
        partial_df <- data.frame(Model = part_call,
                    Pearson = part_pearson,
                    Sig = part_significance
                    )
        out_df = rbind(df,partial_df)
        save_path = sprintf("/data/movie-associations/mantel_results/objtrain_bold_imgnet_brain/%s",layer)
        write.csv(out_df,sprintf("%s/%s_mantel_results.csv",save_path,roi), row.names = FALSE)
    }
}