library("vegan")

layers <- c('conv1','conv2','conv3','conv4','conv5','fc6','fc7')
#layers <- c('conv5')
rdm_folder <- "/data/movie-associations/rdms/main/replic_training"
save_path <- "/data/movie-associations/mantel_results/main_imgnet_lch/replic_training"

n_categories <- 256
# n_categories <- 2000

lch <- read.table("/data/movie-associations/rdms/semantic_models/lch_distance_256-imgnet.txt") 
# bold <- read.csv("/data/movie-associations/rdms/semantic_models/MSCOCO_BOLD5000_2000_cosine_distance.csv")
    
sem_model <- lch
# sem_model <- bold[2:(n_categories+1)]

sem_model <- (sem_model - min(sem_model)) / (max(sem_model) - min(sem_model))

for (layer in layers) {

    rdm_path <- list.files(path=sprintf("%s/%s",rdm_folder,layer), pattern="*.csv", full.names=TRUE, recursive=FALSE)

    print("semantic model read in")

    call <- c()
    pearson  <- c()
    significance  <- c()

    part_call <- c()
    part_pearson  <- c()
    part_significance  <- c()

    for (rdm_file in rdm_path) {
        model <- sapply(strsplit(rdm_file,'/'), `[`, 8)
        model <- sapply(strsplit(model,'_'), `[`, 1)
        print(sprintf("... %s",model))

        if (grepl('Lab', model, fixed = TRUE)) {
            random <- read.csv(sprintf("%s/%s/random-Lab_%s.csv",rdm_folder,layer,layer))
            print("Lab random read in")
        } else if (grepl('supervised', model, fixed = TRUE)) {
            random <- read.csv(sprintf("%s/%s/random-supervised_%s.csv",rdm_folder,layer,layer))
            print("supervised random read in")
        } else {
            random <- read.csv(sprintf("%s/%s/random-distort_%s.csv",rdm_folder,layer,layer))
            print("default distort random read in")
        }
        
        #max min normalise the random model
        random <- random[2:(n_categories+1)]
        random <- (random - min(random)) / (max(random) - min(random))

        rdm <- read.csv(rdm_file)
        print(sprintf("rdm %s read in", rdm_file))

        # do max/min normalisation for the RDMs
        rdm <- rdm[2:(n_categories+1)]
        rdm <- (rdm - min(rdm)) / (max(rdm) - min(rdm))

        #mantel <- mantel(rdm[2:(n_categories+1)], sem_model)
        mantel <- mantel(rdm, sem_model)
        print(sprintf("mantel %s done", rdm_file))
        
        call<-append(call,model)
        pearson<-append(pearson,mantel[3]$statistic)
        significance<-append(significance,mantel[4]$signif)
        
        #partial <- mantel.partial(rdm[2:(n_categories+1)],sem_model,random[2:(n_categories+1)])
        partial <- mantel.partial(rdm,sem_model,random)
        print(sprintf("partial mantel %s done", rdm_file))
        
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
    write.csv(out_df,sprintf("%s/mantel_imgnet_lch_%s.csv", save_path, layer), row.names = FALSE)
}

quit()