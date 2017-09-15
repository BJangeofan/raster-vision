#!/bin/bash

function _term(){
    # Kill any still-living background processes when the script exits
    set +e
    for PROCESS in ${BACKGROUND_PROCESSES[*]}
    do
        kill -0 "${PROCESS}" >/dev/null 2>&1 && kill -TERM "${PROCESS}"
    done
    set -e
}

# This trains a model on a dataset on EC2.

# Parse args
LOCAL=false
while :; do
    case $1 in
        --config-path)
            CONFIG_PATH=$2
            ;;
        --train-id)
            TRAIN_ID=$2
            ;;
        --dataset-id)
            DATASET_ID=$2
            ;;
        --model-id)
            MODEL_ID=$2
            ;;
        --local)
            LOCAL=true
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)
            break
    esac
    shift
    shift
done

echo "CONFIG_PATH     = ${CONFIG_PATH}"
echo "TRAIN_ID        = ${TRAIN_ID}"
echo "DATASET_ID      = ${DATASET_ID}"
echo "MODEL_ID        = ${MODEL_ID}"
echo "LOCAL           = ${LOCAL}"

set -e -x
cd /opt/src/detection

S3_DATASETS=s3://raster-vision/datasets/detection
LOCAL_DATASETS=/opt/data/datasets/detection

S3_TRAIN=s3://raster-vision/results/detection/train
LOCAL_TRAIN=/opt/data/results/detection/train

# how often to sync files to the cloud
SYNC_INTERVAL="10m"
BACKGROUND_PROCESSES=()

cd /opt/src/detection

# sync results of previous run just in case it crashed in the middle of running
if [ "$LOCAL" = false ] ; then
    rm -Rf "${LOCAL_TRAIN:?}/${TRAIN_ID}"
    aws s3 sync "${S3_TRAIN}/${TRAIN_ID}" "${LOCAL_TRAIN}/${TRAIN_ID}"

    # download pre-trained model (to use as starting point) and unzip
    aws s3 cp "${S3_DATASETS}/models/${MODEL_ID}.zip" "${LOCAL_DATASETS}/models/${MODEL_ID}.zip"
    unzip -o "${LOCAL_DATASETS}/models/${MODEL_ID}.zip" -d "${LOCAL_DATASETS}/models/"

    # download training data and unzip
    aws s3 cp "${S3_DATASETS}/${DATASET_ID}.zip" "${LOCAL_DATASETS}/${DATASET_ID}.zip"
    unzip -o "${LOCAL_DATASETS}/${DATASET_ID}.zip" -d "${LOCAL_DATASETS}"

    sync_s3() {
        while true
        do
            aws s3 sync "${LOCAL_TRAIN}/${TRAIN_ID}" "${S3_TRAIN}/${TRAIN_ID}" --delete
            sleep "${SYNC_INTERVAL}"
        done
    }
    sync_s3 &
    BACKGROUND_PROCESSES+=($!)

fi

mkdir -p "${LOCAL_TRAIN}/${TRAIN_ID}"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

python models/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path="${CONFIG_PATH}" \
    --train_dir="${LOCAL_TRAIN}/${TRAIN_ID}/train" &
BACKGROUND_PROCESSES+=($!)

python models/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path="${CONFIG_PATH}" \
    --checkpoint_dir="${LOCAL_TRAIN}/${TRAIN_ID}/train" \
    --eval_dir="${LOCAL_TRAIN}/${TRAIN_ID}/eval" &
BACKGROUND_PROCESSES+=($!)


# https://unix.stackexchange.com/questions/146756/forward-sigterm-to-child-in-bash
# Kill background processes when killed by Docker (i.e. via the Batch console)
# or when the last command exits.

# monitor results using tensorboard app
tensorboard --logdir="${LOCAL_TRAIN}/${TRAIN_ID}"