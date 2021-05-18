#EXP_HOME=/content
DRIVE_HOME=/content/drive/MyDrive/dlgo

export BOARD_SIZE=9

python3 run_endgame.py \
  --verbose=0 \
  --tar_dir="${DRIVE_HOME}/games" \
  --selfplay_dir="${DRIVE_HOME}/selfplay/train" \
  --holdout_dir="${DRIVE_HOME}/selfplay/val" \
  --sgf_dir="${DRIVE_HOME}/selfplay/sgf" \
  --num_readouts=400 \
  --parallel_readouts=32 \
  --load_file="${DRIVE_HOME}/checkpoints/model3_epoch_5.h5"
  2>&1 | tee "${DRIVE_HOME}/run_endgame.log"

