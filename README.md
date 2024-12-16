# Multiomics sex differences in AD

AD trajectory comparisons are done between sexes estimating differences in magnitude, orientation, and shape.

Usage:
- `pls_da`: Partial least squares regression of concatenated metabolomics with AD diagnosis as outcome. A double cross-validation is used to ensure proper estimation of the latent variables avoiding overfitting. It returns a `.csv` table with RID, and latent variable score. Options:
	- `-R`: number of times the cross-validation loop should be repeated. Default 30.
	- `-C`: maximum number of latent variables to evaluate. Default 100.
- `snf`: Similarity Network Fusion algorithm. It returns a `.csv` table with RID and spectral embedding scores.
- `sex_diff`: Sex difference analysis using the latent variables obtained from the PLS-DA analysis. It returns a table with the estimated parameter and p-value from a residual randomization. Options:
	- `-I`: number of iterations to run the randomization. Default 999.
  - `-F`: type of file to use as input for the sex difference analysis. Either `pls` or `snf`. Default `pls`.
- `make_plots`: Will generate three plots. A scatter plot of all coordinates from the scores file file, with groups and diagnoses highlighted. An orientation plot showing the estimated orientation for each group in each coordinate. A Correlation plot from the metabolomics data. Options:
  - `-F`: type of file to use as input for the plots. Either `pls` or `snf`. Only the `pls` option generates a VIP plot.
