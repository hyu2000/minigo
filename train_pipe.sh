#!/bin/bash

set -e  # exit the script if any command fails
export BOARD_SIZE=9

# DRIVE_HOME=/content/drive/MyDrive/dlgo
DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/9x9
PYTHON=/Users/hyu/anaconda/envs/tf2/bin/python3
# PYTHON=echo   # testing
MINIGO=/Users/hyu/PycharmProjects/dlgo/minigo

MODEL_DIR="${DRIVE_HOME}/checkpoints"
EPOCH=2


# main selfplay iterations
for i in {7..11}
do
#  SELFPLAY_DIR="${DRIVE_HOME}/selfplay${i}p"
#  echo "selfplay puzzles: ${SELFPLAY_DIR}"
#  mkdir -p ${SELFPLAY_DIR}
#
#  for worker in {1..3}
#  do
#    ${PYTHON} ${MINIGO}/run_selfplay.py \
#      puzzles \
#      --verbose=0 \
#      --selfplay_dir="${SELFPLAY_DIR}/train" \
#      --holdout_dir="${SELFPLAY_DIR}/val" \
#      --sgf_dir="${SELFPLAY_DIR}/sgf" \
#      --full_readout_prob=1.0 \
#      --parallel_readouts=16 \
#      --holdout_pct=0 \
#      --softpick_move_cutoff=0 \
#      --dirichlet_noise_weight=0.25 \
#      --reduce_symmetry_before_move=0 \
#      --num_readouts=200 \
#      --num_games=300 \
#      --load_file="${MODEL_DIR}/model${i}_${EPOCH}.mlpackage" \
#      --resign_threshold=-1 \
#      2>&1 | tee "${SELFPLAY_DIR}/run_selfplay${worker}.log" &
#    done
#
#  wait
  # if any process stalled, we will not reach here

  SELFPLAY_DIR="${DRIVE_HOME}/selfplay${i}f"
  echo "selfplay games: ${SELFPLAY_DIR}"
  mkdir -p ${SELFPLAY_DIR}

  for worker in {1..3}
  do
    ${PYTHON} ${MINIGO}/run_selfplay.py \
      games \
      --verbose=0 \
      --selfplay_dir="${SELFPLAY_DIR}/train" \
      --holdout_dir="${SELFPLAY_DIR}/val" \
      --sgf_dir="${SELFPLAY_DIR}/sgf" \
      --full_readout_prob=1.0 \
      --parallel_readouts=16 \
      --holdout_pct=0 \
      --softpick_move_cutoff=6 \
      --dirichlet_noise_weight=0.25 \
      --num_readouts=200 \
      --reduce_symmetry_before_move=3 \
      --num_games=1000 \
      --load_file="${MODEL_DIR}/model${i}_${EPOCH}.mlpackage" \
      --resign_threshold=-1 \
      2>&1 | tee "${SELFPLAY_DIR}/run_selfplay${worker}.log" &
    done

  wait

#  if [ $? -ne 0 ]; then
#      echo "run_selfplay ${i} failed"
#      exit 1
#  fi

  for selfplay_type in "f"; do
    SELFPLAY_DIR="${DRIVE_HOME}/selfplay${i}${selfplay_type}"
    ENHANCED_DIR="${DRIVE_HOME}/selfplay${i}${selfplay_type}/enhance"
    ${PYTHON} ${MINIGO}/enhance_ds.py ${SELFPLAY_DIR} ${ENHANCED_DIR} 2>&1 | tee "${SELFPLAY_DIR}/enhance${i}.log"
  done

  # save data to drive?

  TRAIN_LOG_DIR="${DRIVE_HOME}/selfplay${i}f"
  ${PYTHON} ${MINIGO}/run_train.py "${DRIVE_HOME}/selfplay${i}f/enhance/train" ${MODEL_DIR} "${i}_${EPOCH}" 2>&1 | tee "${TRAIN_LOG_DIR}/train${i}.log"

done

echo "Done with train_pipe iter ${i}"