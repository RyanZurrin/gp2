#!/bin/bash -e
DOCKER_IMAGE="58a38fc60eeb"
RUN_MODE="CADt"
PREFIX="DeepSight"
INPUT_DIR=""
OUTPUT_DIR=""

### DISPLAY PRODUCT LABEL
echo " ======================================================================="
echo "||     DeepSight Mammography Research Software                          ||"
echo "||     DeepSight is for research purposes only and not for clinical use ||"
echo "||     DeepHealth, Inc.                                                 ||"
echo "||     Version $DOCKER_IMAGE                                             ||"
echo "||     Copyright © 2021 by DeepHealth, Inc.                             ||"
echo "||     Bringing the Best Doctor in the World to Every Patient           ||"
echo " ======================================================================="
echo $label

### Read script parameters
GPU="" # GPU used -e CUDA_VISIBLE_DEVICES flag
INFO="False" # just display the product label
OPERATE_POINT="" # passed to centaur, for CADt operating points
ADDITIONAL_PARAMS=""
CASELIST=""

while (( "$#" )); do
  case "$1" in
    -gpu|--gpu)
      GPU=$2
      shift
      shift
      ;;
    -i|--input)
      INPUT_DIR=$2
      shift
      shift
      ;;
    -o|--output)
      OUTPUT_DIR=$2
      shift
      shift
      ;;
    -op|--operating_point)
      OPERATE_POINT=$2
      shift
      shift
      ;;
    -info|--information)
      INFO="True"
      shift
      ;;
    -sd|--save_to_disk)
      SAVE_DISK="True"
      shift
      ;;
    -log|--logging_level)
      LOG_LEVEL=$2
      shift
      shift
      ;;
    -log_i|--logging_interval)
      LOG_INTERVAL=$2
      shift
      shift
      ;;
    -log_c|--logging_backup_count)
      LOG_BACKUP_COUNT=$2
      shift
      shift
      ;;
    -cl|--caselist)
      CASELIST=$2
      shift
      shift
      ;;
    --additional_params)
        # Extra Centaur params
        shift
        while (( "$#" )); do
          ADDITIONAL_PARAMS="$ADDITIONAL_PARAMS $1"
          shift
        done
        ;;
    *)
      echo "invalid arguments passed in $1"
      exit 1
    esac
done



# Just display product label?
if [ "$INFO" == "True" ]; then
  exit 0
fi

usage="Usage: start.sh -i|--input_dir INPUT_DIR -o|--output_dir OUTPUT_DIR [-cl|--caselist CASELIST_FILE]"

if [ "$INPUT_DIR" == "" ]; then
  echo $usage
  echo "Please specify a value for 'input_dir' parameter"
  exit 1
fi

if [ "$OUTPUT_DIR" == "" ]; then
  echo $usage
  echo "Please specify a value for 'output_dir' parameter"
  exit 1
fi

if ls -1qA ${OUTPUT_DIR}/ | grep -q .; then
  echo $usage
  echo "Please make sure the output folder is empty (${OUTPUT_DIR})"
  exit 1
fi

# Get a unique id for the docker container
uid=$(python -c 'import uuid; print(str(uuid.uuid1()))')
docker_container_name="${PREFIX}-${uid}"

# Create python command to be run in the container
python_command="python centaur/centaur_deploy/deploy.py --run_mode ${RUN_MODE} --reports most_malignant_image summary --use_heartbeat n --checks_to_ignore FAC-140 FAC-200"

if [ "$SAVE_DISK" == "True" ]; then
   python_command="${python_command} --save_to_disk"
fi

if [ "$OPERATE_POINT" != "" ]; then
   python_command="${python_command} --cadt_operating_point_key ${OPERATE_POINT}"
fi

if [ "$LOG_LEVEL" != "" ]; then
   python_command="${python_command} --logging_level ${LOG_LEVEL}"
fi

if [ "$LOG_INTERVAL" != "" ]; then
   python_command="${python_command} --logging_interval ${LOG_INTERVAL}"
fi

if [ "$LOG_BACKUP_COUNT" != "" ]; then
   python_command="${python_command} --logging_backup_count ${LOG_BACKUP_COUNT}"
fi

if [ "$ADDITIONAL_PARAMS" != "" ]; then
   python_command="${python_command} ${ADDITIONAL_PARAMS}"
fi


if  [ "$CASELIST" == "" ]; then
  ### Regular scenario
  docker_command="docker run --rm \
    -e CUDA_VISIBLE_DEVICES=${GPU} \
    -v ${INPUT_DIR}:/root/input  \
    -v ${OUTPUT_DIR}:/root/output \
    --name ${docker_container_name} \
    ${DOCKER_IMAGE} /bin/bash -c '${python_command}'"
else
  ### Caselist file
  docker_command="docker run --rm \
    -e CUDA_VISIBLE_DEVICES=${GPU} \
    -v ${CASELIST}:/root/caselist.txt  \
    -v ${INPUT_DIR}:${INPUT_DIR} \
    -v ${OUTPUT_DIR}:/root/output \
    --name ${docker_container_name} \
    ${DOCKER_IMAGE} /bin/bash -c '${python_command} --input_dir /root/caselist.txt'"
fi

eval $docker_command

