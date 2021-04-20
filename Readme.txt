NB This is research code. The implementation of MetaGrad is
mathematically correct, but it is in no way optimized, and runs very
slowly because it is written in Python. 

To run the experiments several packages are required:
- pandas
- numpy
- scipy
- cvxpy
- copy
- sklearn
- importlib
- time 
- matplotlib
- cycler
- sys

Datasets are not included. We used the following data sets, which can be
downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

Regression: abalone_scale, bodyfat_scale, cpusmall_scale, housing_scale,
mg_scale, space_ga_scale
Classification: a9a, australian_scale, breast-cancer_scale,
covtype_scale, diabetes_scale, heart_scale, ijcnn1, ionosphere_scale,
phishing, splice_scale, w8a

To run an individual experiment one can run "RunExperiments.py", which 
uses the functions in "RunLibsvm.py" to either compute the coefficients 
or run an optimizer on a dataset.

To run all experiments from the experiments section of the paper, one
can run "make".
To run all the hypertuned experiments from the appendix, one can run
"make hypertunes"
We ran the experiments for multiple data sets in parallel on a multicore
machine, using the -j option of make.

To combine the results from all experiments one can run the files
"combine_results.py" and "hypertune_combine_results.py".

To produce the plots from the paper one can run the files "Plots.R" and 
"hypertune_plot.py".

To produce latex code for the tables from the paper one can run the file 
"csv to latex.R".

To see the implementation of MetaGrad see "LipschitzGrad.py". This file 
includes the following versions of MetaGrad: the full version, the 
frequent directions version, and the diagonal version. 

The file "runoptimizer.py" contains a wrapper for our MetaGrad
implementation as well as for an implemention of AdaGrad and Online
Gradient Descent. This wrapper allows the user to more easily run
individual experiments and provides some plots.

Our implementations of AdaGrad and Online Gradient Descent can be found
in "AdaGrad.py" and "GD.py" respectively. 

The file "customregression.py" contains the code that we use to find
the offline minimizer of the losses.


