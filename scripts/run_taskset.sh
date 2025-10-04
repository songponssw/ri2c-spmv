#!/bin/bash -l
# export DATASETS=/hdd
# Run with ./run_taskset.sh exectime/papi/exectime_omp inputtype csr thread papi_set 


getCurrentTime(){
  date "+%Y%m%d  %H:%M:%S"
}

### Passing arguments
EXP=${1}
MTX=${2}
TARGET_FORMAT=${3}
LOCAL_OMP_NUM=${4}
LOG_FILE="log.log"



# Header default is execution time
header="mtx,outer0,outer1,outer2,outer3,outer4,outer5,outer6,outer7,outer8,outer9,coo2csr,csr2target,inner,outer"
output_csv="output/${MTX}/run_${EXP}-${LOCAL_OMP_NUM}-${TARGET_FORMAT}.csv"

if [[ ${EXP} == "features" ]];then
  header="mtx,N,nnz,min_nnz_row,max_nnz_row,avg_nnz_row,nnz_row_sd,nnz_row_var,empty_row,num_cols,ell_elem,ell_ratio,num_diags,diag_elem,dia_ratio,stride,is_symmetry"
  output_csv="output/${MTX}/features.csv"
  echo "features"
fi

if [[ ${EXP} == "ldist" ]];then
  header="mtx"
  for i in $(seq 1 $LOCAL_OMP_NUM);do
    header+=",th$i"
  done
  output_csv="output/${MTX}/ldist.csv"
fi

if [[ ${EXP} == "row_dist" ]];then
  header="mtx,N,row_dist,col_dist"
  output_csv="output/${MTX}/row_dist.csv"
fi
if [[ ${EXP} == "k_dist" ]];then
  header="mtx,num_diags,k_dist"
  output_csv="output/${MTX}/k_dist.csv"
fi

# Core mapping
if [[ ${LOCAL_OMP_NUM} == '2' ]]; then
  cpu_id="0,1"
elif [[ ${LOCAL_OMP_NUM} == '4' ]]; then
  cpu_id="0-3"
elif [[ ${LOCAL_OMP_NUM} == '8' ]]; then
  cpu_id="0-7"
elif [[ ${LOCAL_OMP_NUM} == '16' ]]; then
  cpu_id="0-15"
elif [[ ${LOCAL_OMP_NUM} == '32' ]]; then
  cpu_id="0-31"
elif [[ ${LOCAL_OMP_NUM} == '64' ]]; then
  cpu_id="0-63"
else # 0 and 1 case
  cpu_id="0"
fi




# Config binfile and output
binfile="run_${EXP}"
echo ${header} > ${output_csv}


# Get matrix
DATASETS="/home/$USER/hdd/benchmarks"
path=$DATASETS
mkdir -p output/${MTX}
readarray mtxfiles < ./scripts/mtx-names/${MTX}.name
count=0

# Logging
touch $LOG_FILE
echo "start ${binfile} ${LOCAL_OMP_NUM} ${TARGET_FORMAT} ${EXP} ${MTX} at $(getCurrentTime)" >> $LOG_FILE
echo "PASS: ${count} mtx" >> $LOG_FILE

# Main Looop
for mtx in ${mtxfiles[@]}
do
  m=$path"/"$mtx
  echo "running "$m"..."
  out1=`OMP_NUM_THREADS=${LOCAL_OMP_NUM} OMP_PROC_BIND=close OMP_PLACES=sockets \
    taskset -c ${cpu_id} ./bin/${binfile} $m ${TARGET_FORMAT} 2>/dev/null`
  if [ $? -ne 0 ]
  then
    out1="${TARGET_FORMAT} error"
  fi

  if [[ $EXP == "papi_omp" ]] || [[ $EXP == "papi" ]]; then
    echo "$out1" >> $output_csv
  else 
    echo "$m,$out1" >> $output_csv
  fi

  ((count++))
  update_content="PASS: ${count} mtx"
  sed -i '$s/.*/'"$update_content"'/' "$LOG_FILE"
done


echo "finish ${binfile} ${LOCAL_OMP_NUM} ${TARGET_FORMAT} at $(getCurrentTime)" >> ${LOG_FILE}
echo "------------\n" >> ${LOG_FILE}

: <<'END'
block comment
END
