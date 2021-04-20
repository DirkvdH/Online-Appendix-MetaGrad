# This makefile will run all our experiments. Its main point is to run them in  parallel on a big machine, with make -j 16.

.DELETE_ON_ERROR:
.PHONY: all clean coefs killers hypertunes

SHELL := bash

# name separator that Dirk eventually picked
SEP := +

# all our algorithms
ALGS := AdaGrad GDn GDt MGFull MGdiag MGF2 MGF11 MGF26 MGF51

# regression data sets and loss functions
REGRESSION_DATA := abalone_scale bodyfat_scale cpusmall_scale housing_scale mg_scale space_ga_scale
REGRESSION_LOSSES := absolute squared

# classification data sets and loss functions
CLASSIFICATION_DATA := a9a australian_scale breast-cancer_scale covtype_scale diabetes_scale heart_scale ijcnn1 ionosphere_scale phishing splice_scale w8a
CLASSIFICATION_LOSSES := hinge logistic


# all the files that we want to obtain
TARGETS := $(foreach alg, $(ALGS), $(foreach set, $(REGRESSION_DATA), $(foreach loss, $(REGRESSION_LOSSES), $(set)$(SEP)$(loss)$(SEP)$(alg).dat)) \
	   $(foreach set, $(CLASSIFICATION_DATA), $(foreach loss, $(CLASSIFICATION_LOSSES), $(set)$(SEP)$(loss)$(SEP)$(alg).dat)))

# the coefficient files (each $(TARGETS) has one such as its dependency)
COEFS := $(foreach set, $(REGRESSION_DATA), $(foreach loss, $(REGRESSION_LOSSES), $(set)$(SEP)$(loss)$(SEP)coef.npy)) \
         $(foreach set, $(CLASSIFICATION_DATA), $(foreach loss, $(CLASSIFICATION_LOSSES), $(set)$(SEP)$(loss)$(SEP)coef.npy))


HYPERTUNES := $(foreach set, $(REGRESSION_DATA), $(foreach loss, $(REGRESSION_LOSSES), hypertune_$(set)$(SEP)$(loss).pdf)) \
              $(foreach set, $(CLASSIFICATION_DATA), $(foreach loss, $(CLASSIFICATION_LOSSES), hypertune_$(set)$(SEP)$(loss).pdf))

# sudo make me all targets
all : $(TARGETS)
coefs : $(COEFS)
hypertunes: $(HYPERTUNES)
killers : killer_oco.pdf killer_optim.pdf

.SECONDEXPANSION:
# targets depend on coef files
$(TARGETS) : %.dat : $$(word 1, $$(subst $(SEP), ,$$*))$(SEP)$$(word 2, $$(subst $(SEP), ,$$*))$(SEP)coef.npy
	{ time OPENBLAS_NUM_THREADS=1 python3 RunExperiments.py $(subst $(SEP), ,$*) ;} &> $*.log

# coef files have no dependency
$(COEFS) : %$(SEP)coef.npy :
	{ time OPENBLAS_NUM_THREADS=1 python3 RunExperiments.py $(subst $(SEP), ,$*) coefonly ;} &> $*.log

$(HYPERTUNES) : hypertune_%.pdf : hypertune_%.csv hypertune_plot.py
	OPENBLAS_NUM_THREADS=1 python3 hypertune_plot.py $(subst $(SEP), ,$*)

$(HYPERTUNES:.pdf=.csv) : hypertune_%.csv : $$(word 1, $$(subst $(SEP), ,$$*))$(SEP)$$(word 2, $$(subst $(SEP), ,$$*))$(SEP)coef.npy
	{ time OPENBLAS_NUM_THREADS=1 python3 hypertune.py $(subst $(SEP), ,$*) ;} &> hypertune_$*.log


killer_oco.pdf killer_optim.pdf : killer_%.pdf : killer.py
	python3 $< $*


clean:
	rm -rf $(TARGETS) $(TARGETS:.dat=.log) $(COEFS) $(COEFS:.npy=.log)
