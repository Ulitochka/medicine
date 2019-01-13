#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`

TRAINER_PATH=`realpath ${SCRIPT_PATH}/`
cd ${TRAINER_PATH}

#python3 -m tools.data_loader
#python3 -m feature_extractors.voc_features
#python3 -m tools.data_splitter
python3 -m experiments.classif_baseline
#python3 -m grammar.main
# python3 -m experiments.nn_base_line