#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`

TRAINER_PATH=`realpath ${SCRIPT_PATH}/`
cd ${TRAINER_PATH}

# python3 -m data_view.symptoms
python3 -m data_view.descriptions