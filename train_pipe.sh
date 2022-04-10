#!/bin/bash

# DRIVE_HOME=/content/drive/MyDrive/dlgo
DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/5x5
PYTHON=/Users/hyu/anaconda/envs/tf2/bin/python3
MINIGO=/Users/hyu/PycharmProjects/dlgo/minigo

export BOARD_SIZE=5

MODEL_DIR="${DRIVE_HOME}/checkpoints"


# main selfplay iterations
for i in {8..8}
do
  SELFPLAY_DIR="${DRIVE_HOME}/selfplay${i}"
  ENHANCED_DIR="${DRIVE_HOME}/selfplay${i}/enhance"
  echo "selfplay: ${SELFPLAY_DIR}"

  # rm -rf ${SELFPLAY_DIR}
  mkdir -p ${SELFPLAY_DIR}

  for worker in {1..4}
  do
    ${PYTHON} ${MINIGO}/run_selfplay.py \
      --verbose=0 \
      --selfplay_dir="${SELFPLAY_DIR}/train" \
      --holdout_dir="${SELFPLAY_DIR}/val" \
      --sgf_dir="${SELFPLAY_DIR}/sgf" \
      --softpick_move_cutoff=6 \
      --dirichlet_noise_weight=0.25 \
      --num_readouts=200 \
      --full_readout_prob=1.0 \
      --reduce_symmetry_before_move=8 \
      --parallel_readouts=16 \
      --holdout_pct=0 \
      --num_games=300 \
      --load_file="${MODEL_DIR}/model${i}_epoch2.h5" \
      --resign_threshold=-1 \
      2>&1 | tee "${SELFPLAY_DIR}/run_selfplay${worker}.log" &
    done

  wait

  if [ $? -ne 0 ]; then
      echo "run_selfplay ${i} failed"
      break
  fi

  ${PYTHON} ${MINIGO}/enhance_ds.py ${SELFPLAY_DIR} ${ENHANCED_DIR} 2>&1 | tee "${SELFPLAY_DIR}/enhance${i}.log"

  # save data to drive?
  if [ $? -ne 0 ]; then
      echo "enhance_ds ${SELFPLAY_DIR} failed"
      break
  fi

  ${PYTHON} ${MINIGO}/run_train.py "${ENHANCED_DIR}/train" ${MODEL_DIR} $i 2>&1 | tee "${SELFPLAY_DIR}/train${i}.log"

  if [ $? -ne 0 ]; then
      echo "run_train ${i} failed"
      break
  fi

done
