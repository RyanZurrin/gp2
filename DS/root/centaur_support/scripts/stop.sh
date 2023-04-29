#!/bin/bash -e

function getJsonVal() {
    # Read a key value from the config.json file (asume that $config_file has been initialized properly)
    key="$1"
    value=`cat $config_file | grep $key | cut -f2- -d ":"`
    value=`echo "${value//\"/}"`
    value=`echo "${value//,/}"`
    value="$(echo -e "${value}" | tr -d '[:space:]')"
    echo $value
}

current_dir=`readlink -e $(dirname "$0")`
config_file="${current_dir}/config.json"
if [ ! -f "$config_file" ]; then
    echo "Config file could not be found in ${config_file}. Please contact DeepHealth for assistance"
    exit 1
fi

# Constant for easy future modification
PREFIX="SAIGE-Q-" # docker container prefix
SHOW_ALL_CONTAINERS_MESSAGE="show" # what user input to show all running containers.
STOP_ALL_CONTAINERS_MESSAGE="all" # what user input to stop all running containers

# default path
SUPPORT_DIR="$(getJsonVal "SUPPORT_DIR")"
EXECUTION_LOG="${SUPPORT_DIR}/execution_log.txt"

# initiate EXECUTION_ID to None
EXECUTION_ID="NONE"

# display all running execution set to false
SHOW_ALL_CONTAINERS="false"


# usage:
__usage="

==============================================================================================================================
# USAGE: Stop execution of SAIGE-Q-X containers.                                                                             #
# stop.sh -option1 value1 ... -optionX valueX                                                                                #
# Options:                                                                                                                   #
#  show       --show                  Shows all running EXECUTION_ID                                                         #
#   -eid *    --execution_id *        Specifies with ExecutionID to stop, eg -eid 000001. 'all' stops all the containers     #
#   -log *    --log_path *            Specifics path in which the log is stored, default value exists, eg -log execution.log #
==============================================================================================================================
"

# reading arguments passed in
while (( "$#" )); do
  case "$1" in
    -eid|--execution_id)
      EXECUTION_ID=$2
      shift
      shift
      ;;
    -log|--log_path)
      EXECUTION_LOG=$2
      shift
      shift
      ;;
    ${SHOW_ALL_CONTAINERS_MESSAGE}|--${SHOW_ALL_CONTAINERS_MESSAGE})
      SHOW_ALL_CONTAINERS="true"
      shift
      ;;

    *)
      # display usage message
      echo "$__usage"
      exit 1

  esac
done


# Display running execution IDs and exit
if [ "$SHOW_ALL_CONTAINERS" == "true" ]; then
  echo "Running Execution ID(s):"
  echo $(docker ps --filter "name=${PREFIX}*" --format "{{.Names}}" | tr -d ${PREFIX})
  exit 0
fi


# log path doesnt exist exit with error and terminate
if [ ! -f "${EXECUTION_LOG}" ]; then
  echo "execution log path ${EXECUTION_LOG} does not exist"
  exit 1
fi


if [ "${EXECUTION_ID}" == "NONE" ]; then
  echo "$__usage"
  exit 1
fi

if [ "${EXECUTION_ID}" == "all" ]; then
  echo "stopping all running containers"
  # ids for logging
  ids="$(docker ps --filter "name=${PREFIX}*" --format "{{.Names}}" | sed 's/'${PREFIX}'//' ) "

else
  echo "stopping execution id ${EXECUTION_ID} "
  ids="${EXECUTION_ID}"
fi

for id in $ids; do
  echo "Stopping ${id}..."
  # Kill container
  docker kill ${PREFIX}${id}
  echo "Docker container stopped (${PREFIX}${id})"
  # Kill monitoring service
  monitoring_pid=`ps -eo pid,cmd | grep /scripts/monitoring_service/service.py | grep $id | sed -e 's/^[[:space:]]*//' | cut -f1 -d ' '`
  echo "Stopping process ${monitoring_pid}..."
  kill $monitoring_pid
  echo "Monitoring service stopped (${monitoring_pid})"
  # Log
  TIME="$(date -u '+%Y%m%d%H%M%S')"
  LOG_MESSAGE="STOP: ExecutionID $id stopped on $TIME "
  echo "${LOG_MESSAGE}" | tee -a $EXECUTION_LOG
done
