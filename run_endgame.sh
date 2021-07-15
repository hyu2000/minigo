#EXP_HOME=/content
DRIVE_HOME=/content/drive/MyDrive/dlgo

export BOARD_SIZE=9

python3 run_endgame.py \
  --verbose=0 \
  --tar_dir="${DRIVE_HOME}/games" \
  --selfplay_dir="${DRIVE_HOME}/selfplay/train" \
  --holdout_dir="${DRIVE_HOME}/selfplay/val" \
  --sgf_dir="${DRIVE_HOME}/selfplay/sgf" \
  --num_readouts=200 \
  --dirichlet_noise_weight=0.025 \
  --parallel_readouts=16 \
  --load_file="${DRIVE_HOME}/checkpoints/endgame1_epoch_2.h5" \
  2>&1 | tee "${DRIVE_HOME}/run_endgame.log"
