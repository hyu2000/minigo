DRIVE_HOME=/content/drive/MyDrive/dlgo
#DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/5x5

export BOARD_SIZE=5

LOCAL_HOME=/tmp
MODEL_DIR="${DRIVE_HOME}/checkpoints"
LOG_DIR="${DRIVE_HOME}/logs"

# bash 3 supports range
for i in {0..0}
do
  SELFPLAY_DIR="${LOCAL_HOME}/selfplay${i}"
  echo "selfplay: ${SELFPLAY_DIR}"

  rm -rf ${SELFPLAY_DIR}
  mkdir -p ${SELFPLAY_DIR}

  python3 run_selfplay.py \
  --verbose=0 \
  --selfplay_dir="${SELFPLAY_DIR}/train" \
  --holdout_dir="${SELFPLAY_DIR}/val" \
  --sgf_dir="${SELFPLAY_DIR}/sgf" \
  --holdout_pct=0 \
  --softpick_move_cutoff=20 \
  --dirichlet_noise_weight=0.25 \
  --num_games_share_tree=1 \
  --num_readouts=200 \
  --parallel_readouts=16 \
  --num_games=3000 \
  2>&1 | tee "${LOG_DIR}/selfplay${i}.log"

  #  --load_file="${MODEL_DIR}/model${i}_epoch1.h5" \

  if [ $? -ne 0 ]; then
      echo "run_selfplay ${i} failed"
      break
  fi

  python3 enhance_ds.py ${SELFPLAY_DIR} 2>&1 | tee "${LOG_DIR}/enhance${i}.log"

  # save data to drive?
  if [ $? -ne 0 ]; then
      echo "enhance_ds ${SELFPLAY_DIR} failed"
      break
  fi

  python3 run_train.py "${SELFPLAY_DIR}/train" ${MODEL_DIR} $i 2>&1 | tee "${LOG_DIR}/train${i}.log"

  if [ $? -ne 0 ]; then
      echo "run_train ${i} failed"
      break
  fi

done

