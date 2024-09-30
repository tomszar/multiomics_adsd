library(SNFtool)

K <- 20
# number of neighbors, usually (10~30)
alpha <- 0.5
# hyperparameter, usually (0.3~0.8)
T <- 10
# Number of Iterations, usually (10~20)

p180 <- read.csv("P180.csv", row.names = "RID")
nmr <- read.csv("NMR.csv", row.names = "RID")
common_rows <- intersect(row.names(p180), row.names(nmr))
p180 <- p180[common_rows, ]
nmr <- nmr[common_rows, ]
p180 <- standardNormalization(p180)
nmr <- standardNormalization(nmr)
Dist1 <- dist2(as.matrix(p180), as.matrix(p180))
Dist2 <- dist2(as.matrix(nmr), as.matrix(nmr))
W1 <- affinityMatrix(Dist1, K, alpha)
W2 <- affinityMatrix(Dist2, K, alpha)
W <- SNF(list(W1, W2), K, T)
