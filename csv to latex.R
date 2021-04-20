library(xtable)
overview <- read.csv("overview datasets.csv")

tolerance = 1; # proximity to best to share podium

# use only suitable datasets
used_datasets <- c("abalone_scale", "bodyfat_scale", "cpusmall_scale", "housing_scale", "mg_scale", "space_ga_scale",
                   "a9a", "australian_scale", "breast-cancer_scale", "covtype_scale", "diabetes_scale", "heart_scale", 
                   "ijcnn1", "ionosphere_scale", "phishing", "splice_scale", "w1atest", "w8a")

overview <- overview[overview$Dataset %in% used_datasets, ]
overview <- overview[, -c(4, 5, 8, 9)]
overview$Dataset <- gsub("_scale", "", overview$Dataset)
colnames(overview) <- c("Dataset", "T", "d", "Outcome", "P(y = 1)")
overview <- overview[order(overview$Outcome), ]

print(xtable(overview), include.rownames = FALSE)

mktables <- function(filename, comparator) {
results <- read.csv(filename)
results$data <- gsub("_scale", "", results$data)
colnames(results)[c(1, 2)] = c("Dataset", "Loss")
# colnames(results)[6:ncol(results)] <- c("MGCo", "MGF2",   "MGF11",  "MGF26",  "MGF51",  "MGFull")
outorder <- order(ifelse(results$Loss %in% c("hinge", "logistic"), "binary", "real"))
results <- results[outorder, ]
textresults <- lapply(1:nrow(results), function(i){
  row <- results[i, 3:ncol(results)]
  sel <- row - min(row, na.rm = T) < tolerance
  chrow <- as.character(round(row, 0))
  chrow[sel] <- paste0("BOLD", chrow[sel])
  return(chrow)})
myres <- data.frame(cbind(results[, 1:2], do.call(rbind, textresults)))
colnames(myres) <- colnames(results)
# remove second occurrence of each data set name
myres$Dataset[(1:(nrow(myres)/2))*2] <- ''
# use multirow to center dataset name across two rows (looks confusing)
#myres$Dataset[(1:(nrow(myres)/2))*2-1] <- paste0('\\multirow{2}{*}{',myres$Dataset[(1:(nrow(myres)/2))*2-1],'}')

boldfilter <- function(x) gsub('BOLD(.*)', paste('\\\\textbf{\\1','}', sep = ""), x)
myxtable <- xtable(myres, caption  = paste("The regret of each algorithm for the various datasets and loss functions. Boldface indicates the regret differs less than $", tolerance, "$ from the minimum regret for that dataset and loss combination",sep=""), 
                   label = "tab:regret table")
print(myxtable, include.rownames = FALSE, sanitize.text.function = boldfilter)


regrets <- results[, 3:ncol(results)]
regretComparator <- results[, comparator, drop = T]
ClassGDt <- regrets / regretComparator

Classmn <- colMeans(ClassGDt, na.rm = T)
classmed <- sapply(ClassGDt, median, na.rm = T)
classvar <- sapply(ClassGDt, sd, na.rm = T)
Ndatasets <- rep(nrow(ClassGDt), length(colnames(results)[-c(1,2)]))
Nbest <- colSums(do.call(rbind, lapply(1:nrow(ClassGDt), function(i){regrets[i, ] - min(regrets[i, ], na.rm = T) < tolerance})), na.rm = T)
NbetterGDt <- colSums(do.call(rbind, lapply(1:nrow(ClassGDt), function(i) regrets[i, ] < regretComparator[i] + tolerance)), na.rm = T)
sumry <- data.frame(colnames(results)[- c(1, 2)], Nbest, NbetterGDt, Classmn, classmed, classvar)
colnames(sumry) <- c("Algorithm", "Nbest", "Nbetter", "MeanRatio", "MedianRatio", "SdRatio")

print(xtable(sumry[c("Algorithm", "Nbest", "Nbetter", "MedianRatio")], digits = c(0,0,0,0,2)), include.rownames = FALSE)

#best <- colnames(results)[3:ncol(results)][]
#cbind(results[, 1:2], best)
}




#mktables("results.csv", "GDt")
mktables("hypertune_results.csv", "AdaGrad")
