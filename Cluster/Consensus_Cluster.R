
# lib ---------------------------------------------------------------------

# install.packages("ConsensusClusterPlus")
library(ConsensusClusterPlus)
library(tidyverse)


# cluster -----------------------------------------------------------------

lgg_dr <- read_csv("LGG_dr.csv")

path_pdf <- "~/cluster_pdf"

result <- ConsensusClusterPlus(
  data.matrix(t(lgg_dr)),
  maxK = 6,
  reps = 1000,
  pItem = 0.8,
  pFeature = 1,
  plot = "pdf",
  title = path_pdf,
  distance = "euclidean",
  clusterAlg = "km",
  seed = 2023110400
)

label <- tibble(
  label2 = result[[2]][["consensusClass"]],
  label3 = result[[3]][["consensusClass"]],
  label4 = result[[4]][["consensusClass"]],
  label5 = result[[5]][["consensusClass"]],
  label6 = result[[6]][["consensusClass"]]
)


