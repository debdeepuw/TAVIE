# data preprocessing
# -------------------------------------------------------------------------
# STArMAP dataset: https://lce.biohpc.swmed.edu/star/explorer.php
# Primary Visual Cortex 160 genes
# -------------------------------------------------------------------------

# setting working directory
setwd("~/OneDrive - Texas A&M University/TAVIE/tavie-jupyter-cluster/data/STArMAP")

# reading the gene-expression counts
count = read.csv("dark_replicate_1.count.csv")

# reading the gene names
gene_names = read.csv("dark_replicate_1.gene_symbol.csv")

# setting the gene names to count
colnames(count) = gene_names$x

# reading the spatial locations
spatial_loc = read.csv("dark_replicate_1.loc.csv")
colnames(spatial_loc) = c("x", "y")

data_list = list(count = count, spatial_location = spatial_loc)

saveRDS(data_list, "STArMAP_Data.rds")
