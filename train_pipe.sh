DRIVE_HOME=/content/drive/MyDrive/dlgo
#DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/5x5

export BOARD_SIZE=5

LOCAL_HOME=/tmp
MODEL_DIR="${DRIVE_HOME}/checkpoints"
LOG_DIR="${DRIVE_HOME}/logs"

# bash 3 supports range
for i in {1..3}
do
  SELFPLAY_DIR="${LOCAL_HOME}/selfplay${i}"
  echo "selfplay: ${SELFPLAY_DIR}"

  mkdir -p "${SELFPLAY_DIR}"

  python3 run_selfplay.py \
  --verbose=0 \
  --selfplay_dir="${SELFPLAY_DIR}/train" \
  --holdout_dir="${SELFPLAY_DIR}/val" \
  --sgf_dir="${SELFPLAY_DIR}/sgf" \
  --holdout_pct=0 \
  --load_file="${MODEL_DIR}/model${i}_epoch2.h5" \
  --softpick_move_cutoff=6 \
  --num_readouts=200 \
  --parallel_readouts=16 \
  --num_games=2000 \
  2>&1 | tee "${LOG_DIR}/selfplay${i}.log"

  if [ $? -ne 0 ]; then
      break
  fi

  python3 enhance_ds.py ${SELFPLAY_DIR} > "${LOG_DIR}/enhance${i}.log" 2>&1

  if [ $? -ne 0 ]; then
      break
  fi

  python3 run_train.py "${SELFPLAY_DIR}/train" ${MODEL_DIR} $i > "${LOG_DIR}/train${i}.log" 2>&1

  if [ $? -ne 0 ]; then
      break
  fi

done

