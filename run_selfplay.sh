DRIVE_HOME=/content/drive/MyDrive/dlgo
#DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/5x5

export BOARD_SIZE=5

mkdir ${DRIVE_HOME}/selfplay

python3 run_selfplay.py \
  --verbose=0 \
  --selfplay_dir="${DRIVE_HOME}/selfplay/train" \
  --holdout_dir="${DRIVE_HOME}/selfplay/val" \
  --sgf_dir="${DRIVE_HOME}/selfplay/sgf" \
  --load_file="${DRIVE_HOME}/checkpoints/model4_epoch_1.h5" \
  --num_readouts=200 \
  --parallel_readouts=16 \
  --num_games=2000 \
  2>&1 | tee "${DRIVE_HOME}/selfplay/run_selfplay.log"

#  --dirichlet_noise_weight=0.025 \
