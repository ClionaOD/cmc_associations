library(‘vegan’)

lch <- read.table("./results/lch_df.txt")
random <- read.csv('./rdms/blurring/conv5/random_rdm_blur10_conv5.csv')

files <- list.files(path="./rdms/blurring/conv5", pattern="*.csv", full.names=TRUE, recursive=FALSE)

get_mantel <- function(rdm, lch, control) {
   mantel <- mantel(rdm[2:257], lch)
   partial <- mantel.partial(rdm[2:257],lch,control[2:257])
   return(mantel, partial)
}

lapply(files, get_mantel(x) {
    rdm <- read.csv(x, header=TRUE) # load file
    # apply function
    list[mantel, partial] <- get_mantel(rdm,lch, random)
    # write to file
    write.table(out, "path/to/output", sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)
})