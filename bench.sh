#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CURR_DIR=$(pwd)

CONTAINER_NAME="enidrift"
CONTAINER_WORKDIR="/opt/enidrift"
ENIDRIFT_SCRIPT="throughput.py"

NUM_PKTS=100000
SAMPLING=2048
RELEASE_SPEED=1000

if [ "$#" -ne 2 ]; then
	echo "Usage: $0 <pcap> <attack-flows>"
	exit 1
fi

PCAP_PATH=$(realpath $1)
PCAP_FILENAME=$(basename -- "$PCAP_PATH")
ATTACK_FLOWS_PATH=$(realpath $2)
ATTACK_FLOWS_FILENAME=$(basename -- "$ATTACK_FLOWS_PATH")
ATTACK="${PCAP_FILENAME%.*}"
CONTAINER_PCAP_PATH="/$PCAP_FILENAME"
CONTAINER_ATTACK_FLOWS_PATH="/$ATTACK_FLOWS_FILENAME"
RESULTS_FILE="$ATTACK.csv"

if ! test -f "$PCAP_PATH"; then
	echo "$PCAP_PATH not found."
	exit 1
fi

if ! test -f "$ATTACK_FLOWS_PATH"; then
	echo "$ATTACK_FLOWS_PATH not found."
	exit 1
fi

assert_file() {
	f=$1
	if ! test -f "$f"; then
		echo "$f not found."
		exit 1
	fi
}

cd $SCRIPT_DIR
cmd=$(cat << EOF
	touch $RESULTS_FILE && \
	python3 $ENIDRIFT_SCRIPT \
		--pcap $CONTAINER_PCAP_PATH \
		--attack $ATTACK \
		--sampling $SAMPLING \
		--release_speed $RELEASE_SPEED \
		--attack_flows $CONTAINER_ATTACK_FLOWS_PATH
EOF
)

docker build . -t $CONTAINER_NAME

# Running with --privileged to disable all security features
# and allow for maximum performance (we can actually detect a
# measurable performance difference with and without this flag).

touch $CURR_DIR/$RESULTS_FILE

docker run \
	--privileged \
	--rm \
	-v "$PCAP_PATH":"$CONTAINER_PCAP_PATH" \
	-v "$ATTACK_FLOWS_PATH":"$CONTAINER_ATTACK_FLOWS_PATH" \
	-v "$CURR_DIR/$RESULTS_FILE":"$CONTAINER_WORKDIR/$RESULTS_FILE" \
	$CONTAINER_NAME \
	/bin/bash -c "export OPENBLAS_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && $cmd"
