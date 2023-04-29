#!/bin/bash -e

# Save original path to restore it once the script ends
OLD_PATH=$PATH

function cleanup {
    # Function that will be always invoked to make sure the PATH is restored
    PATH=OLD_PATH
}

function getJsonVal() {
    # Read a key value from the config.json file (asume that $config_file has been initialized properly)
    key="$1"
    value=`cat $config_file | grep $key | cut -f2- -d ":"`
    value=`echo "${value//\"/}"`
    value=`echo "${value//,/}"`
    value="$(echo -e "${value}" | tr -d '[:space:]')"
    echo $value
}

### Read config file
current_dir=`readlink -e $(dirname "$0")`
config_file="${current_dir}/config.json"
if [ ! -f "$config_file" ]; then
    echo "Config file could not be found in ${config_file}. Please contact DeepHealth for assistance"
    exit 1
fi

### Read script parameters
GPU=0 # GPU used -e CUDA_VISIBLE_DEVICES flag
INFO="False" # just display the product label
CONFIG="False" # display the config file
SAVE_DISK="False" # save to disk flag inside centaur
OPERATE_POINT="None" # passed to centaur, for CADt operating points
REMOVE_INPUT="False" # passed to centaur, for deleting input files
REDIRECT_OUTPUT="False" # Redirect Centaur output to an external file
PREFIX="SAIGE-Q-"
PYTHON_BIN_PATH=""
LOG_LEVEL=""   # Log level in Centaur

while (( "$#" )); do
  case "$1" in
    -gpu|--gpu)
      GPU=$2
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
    -config|--config_file)
      CONFIG="True"
      shift
      ;;
    -save_to_disk|--save_to_disk)
      SAVE_DISK="True"
      shift
      ;;
    -remove_input_files|--remove_input_files)
      REMOVE_INPUT="True"
      shift
      ;;
    -red|--redirect_output)
      REDIRECT_OUTPUT="True"
      shift
      ;;
    -log|--log_level)
      LOG_LEVEL=$2
      shift
      shift
      ;;
    -python_bin|--python_bin_path)
      PYTHON_BIN_PATH=$2
      shift
      shift
      ;;
    *)
      echo "invalid arguments passed in $1"
      exit 1
    esac
done

# Read settings from Config file
DIR_LISTENER_MODE="$(getJsonVal "DIR_LISTENER_MODE")"
CLIENT_DIR="$(getJsonVal "CLIENT_DIR")"
SUPPORT_DIR="$(getJsonVal "SUPPORT_DIR")"
DOCKER_IMAGE="$(getJsonVal "DOCKER_IMAGE")"
CLIENT_INPUT_DIR="$(getJsonVal "CLIENT_INPUT_DIR")"
CLIENT_OUTPUT_DIR="$(getJsonVal "CLIENT_OUTPUT_DIR")"
PACS_RECEIVE_PORT="$(getJsonVal "PACS_RECEIVE_PORT")"
EXECUTION_LOG="${SUPPORT_DIR}/execution_log.txt"

if [ "$PYTHON_BIN_PATH" == "" ]; then
  PYTHON_BIN_PATH="$SUPPORT_DIR/miniconda/bin"
fi

# Capture EXIT signal to restore the original path before exiting
trap cleanup EXIT
# Use the installed Miniconda
PATH=$PYTHON_BIN_PATH:$PATH

### SHA verification
python ${SUPPORT_DIR}/scripts/hash_verifier/verifier.py \
-j ${SUPPORT_DIR}/sha.json \
-e ${SUPPORT_DIR}/sha_errors.txt \
-d ${DOCKER_IMAGE}

### DISPLAY PRODUCT LABEL
label=`docker run -ti --rm ${DOCKER_IMAGE} /bin/bash -c 'python centaur/centaur_deploy/deploy.py --product_label Saige-Q 2> /dev/null'` \
       || (echo "Corrupt installation. Please contact DeepHealth for assistance"; exit 1)
echo "${label}"

# Just display product label?
if [ "$INFO" == "True" ]; then
  exit 0
fi

# Just display client Config?
if [ "$CONFIG" == "True" ]; then
   config_file="${CLIENT_DIR}/config.json"
   python -m json.tool ${config_file}
  exit 0
fi

# Get a unique id for the docker container
uid=$(python -c 'import uuid; print(str(uuid.uuid1()))')
docker_container_name="${PREFIX}${uid}"

# Create python command to be run in the container
python_command="python centaur/centaur_deploy/deploy.py --run_mode CADt"

if [ "$DIR_LISTENER_MODE" == "True" ]; then
  python_command="${python_command} -mondir"
else
  python_command="${python_command} --client_root_path /root/client_dir"
fi

if [ "$SAVE_DISK" == "True" ]; then
   python_command="${python_command} --save_to_disk"
fi

if [ "$REMOVE_INPUT" == "True" ]; then
   python_command="${python_command} --remove_input_files"
fi

if [ "$OPERATE_POINT" != "None" ]; then
   python_command="${python_command} --cadt_operating_point_key ${OPERATE_POINT}"
fi

if [ "$LOG_LEVEL" != "" ]; then
   python_command="${python_command} --logging ${LOG_LEVEL}"
fi


### Start docker container
if [ "$DIR_LISTENER_MODE" == "True" ]; then
  # Check that the output folder is empty
  if ls -1qA $CLIENT_OUTPUT_DIR/ | grep -q .; then
    echo "Output folder ($CLIENT_OUTPUT_DIR) is not empty. Please empty the folder and try again"
    exit 1
  fi

  docker_command="docker run -dt --runtime=nvidia \
  --restart unless-stopped \
  -e CUDA_VISIBLE_DEVICES=${GPU} \
  -v ${CLIENT_INPUT_DIR}:/root/input  \
  -v ${CLIENT_OUTPUT_DIR}:/root/output \
  --name ${docker_container_name} \
  ${DOCKER_IMAGE}"
else
  mount_input_dir="${CLIENT_INPUT_DIR}/${docker_container_name}"
  mount_output_dir="${CLIENT_OUTPUT_DIR}/${docker_container_name}"
  mkdir -p $mount_input_dir
  mkdir -p $mount_output_dir

  docker_command="docker run -dt --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=${GPU} \
  -p ${PACS_RECEIVE_PORT}:${PACS_RECEIVE_PORT} \
  --name ${docker_container_name} \
  -v ${mount_input_dir}:/root/input \
  -v ${mount_output_dir}:/root/output \
  -v ${CLIENT_DIR}:/root/client_dir \
  ${DOCKER_IMAGE}"
fi

eval $docker_command
echo "Docker container created"

TIME="$(date -u '+%Y%m%d%H%M%S')"
if [ "$REDIRECT_OUTPUT" == "True" ]; then
  mkdir -p ${SUPPORT_DIR}/log
  echo "Redirecting output to ${SUPPORT_DIR}/log/$TIME.err"
  docker exec $docker_container_name ${python_command} > ${SUPPORT_DIR}/log/$TIME.err 2>&1 &
else
  docker exec -d $docker_container_name ${python_command}
fi

echo "Saige-Q running..."

### Monitoring service
TIMEOUT=600
MONITOR_FREQ=60
HEARTBEAT_LOG_PATH="$CLIENT_OUTPUT_DIR/heartbeat.log"

nohup python $SUPPORT_DIR/scripts/monitoring_service/service.py -hf ${HEARTBEAT_LOG_PATH} \
-ef ${EXECUTION_LOG} -id ${docker_container_name} \
-t ${TIMEOUT} -fr ${MONITOR_FREQ} > $SUPPORT_DIR/monitoring_service.out 2>&1 &
echo "Monitoring service started"

### Log execution
if [ $? -eq 0 ]; then
  message="START: ExecutionID ${uid} started on ${TIME} \n"
  echo "${message}" | tee -a $EXECUTION_LOG
fi
