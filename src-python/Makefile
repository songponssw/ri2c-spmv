# This directory create .pkl files from .csv
#
dataset=$(word 2, $(MAKECMDGOALS))

all: ft features perf-metric sub-matrices k-dist class_feature merge

# Preprocessing execution time in format-thread config
ft:
	python3 process_ft.py $(dataset)

features:
	python3 process_features.py $(dataset)

perf-metric:
	python3 process_perf_metric.py $(dataset)

sub-matrices:
	python3 process_submatrices.py $(dataset)

k-dist:
	python3 process_k_dist.py $(dataset)

class_feature:
	python3 class_feature.py $(dataset)


merge:
	echo "Merge all pkls to create ..."
	python3 merge_pkl.py $(dataset)
%:
	@:


