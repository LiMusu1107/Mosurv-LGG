
# lib ---------------------------------------------------------------------
library(tidyverse)
library(randomForest)


# RF ----------------------------------------------------------------------

set.seed(2023110400)

forest_mrna <- randomForest(
  x = mrna[, colnames(mrna) != "label2"],
  y = mrna$label2,
  importance = TRUE
)

forest_meth <- randomForest(
  x = meth[, colnames(meth) != "label2"],
  y = meth$label2,
  importance = TRUE
)

forest_mirna <- randomForest(
  x = mirna[, colnames(mirna) != "label2"],
  y = mirna$label2,
  importance = TRUE
)

importance_mrna <- importance(forest_mrna)
importance_meth <- importance(forest_meth)
importance_mirna <- importance(forest_mirna)

