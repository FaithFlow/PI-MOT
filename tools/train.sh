#!/usr/bin/env bash

CONFIG_FILE=$1
PY_ARGS=${@:2}

if [ -z "$CONFIG_FILE" ]; then
  echo "Usage: $0 <config.args> [extra args...]"
  exit 1
fi

OUTPUT_BASE=$(echo "$CONFIG_FILE" | sed -e "s/configs/exps/g" | sed -e "s/.args$//g")
mkdir -p "$OUTPUT_BASE"

for RUN in $(seq 100); do
  ls "$OUTPUT_BASE" | grep "run$RUN" && continue
  OUTPUT_DIR=$OUTPUT_BASE/run$RUN
  mkdir "$OUTPUT_DIR" && break
done

if [ -z "$OUTPUT_DIR" ]; then
  echo "Failed to create a new run directory under $OUTPUT_BASE"
  exit 1
fi

rmpyc() {
  rm -rf $(find -name __pycache__)
  rm -rf $(find -name "*.pyc")
}

echo "Backing up to log dir: $OUTPUT_DIR"
rmpyc && cp -r models datasets util tools main.py engine.py evaluate.py "$CONFIG_FILE" "$OUTPUT_DIR"
echo " ...Done"

cleanup() {
  echo "Packing source code"
  rmpyc
  echo " ...Done"
}

args=$(cat "$CONFIG_FILE")

pushd "$OUTPUT_DIR" || exit 1
trap cleanup EXIT

echo "Logging git status"
git status > git_status
git rev-parse HEAD > git_tag
git diff > git_diff
echo "$PY_ARGS" > desc
echo " ...Done"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-50268}


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m torch.distributed.launch \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  --use_env main.py ${args} --output_dir "$OUTPUT_DIR" |& tee -a output.log
