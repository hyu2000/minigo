DRIVE_HOME=/content/drive/MyDrive/dlgo
#DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/9x9

export BOARD_SIZE=9

LOCAL_HOME=/tmp
MODEL_DIR="${DRIVE_HOME}/checkpoints"
LOG_DIR="${DRIVE_HOME}/logs"
SGF_DIR="${DRIVE_HOME}/sgfs"
TFRECORDS_DIR="${DRIVE_HOME}/tfrecords"

# bash 3 supports range
for i in {3..4}
do
  SELFPLAY_DIR="${LOCAL_HOME}/selfplay${i}"
  echo "selfplay: ${SELFPLAY_DIR}"

  rm -rf ${SELFPLAY_DIR}
  mkdir -p ${SELFPLAY_DIR}

  python3 run_selfplay.py \
  --verbose=0 \
  --load_file="${MODEL_DIR}/model${i}_epoch2.h5" \
  --selfplay_dir="${SELFPLAY_DIR}/train" \
  --holdout_dir="${SELFPLAY_DIR}/val" \
  --sgf_dir="${SGF_DIR}/sgf${i}" \
  --holdout_pct=0 \
  --softpick_move_cutoff=6 \
  --dirichlet_noise_weight=0.25 \
  --num_games_share_tree=1 \
  --num_readouts=200 \
  --parallel_readouts=16 \
  --num_games=1000 \
  2>&1 | tee "${LOG_DIR}/selfplay${i}.log"


  if [ $? -ne 0 ]; then
      echo "run_selfplay ${i} failed"
      break
  fi

  SAMPLES_DIR="${TFRECORDS_DIR}/enhance${i}"
  python3 enhance_ds.py ${SELFPLAY_DIR} ${SAMPLES_DIR} 2>&1 | tee "${LOG_DIR}/enhance${i}.log"

  # save data to drive?
  if [ $? -ne 0 ]; then
      echo "enhance_ds ${SELFPLAY_DIR} failed"
      break
  fi

  python3 run_train.py "${SAMPLES_DIR}/train" ${MODEL_DIR} $i 2>&1 | tee "${LOG_DIR}/train${i}.log"

  if [ $? -ne 0 ]; then
      echo "run_train ${i} failed"
      break
  fi

done

