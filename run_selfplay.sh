DRIVE_HOME=/content/drive/MyDrive/dlgo
DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/9x9

export BOARD_SIZE=9

#mkdir ${DRIVE_HOME}/selfplay

python3 run_selfplay.py \
  --verbose=0 \
  --selfplay_dir="${DRIVE_HOME}/selfplay/train" \
  --holdout_dir="${DRIVE_HOME}/selfplay/val" \
  --sgf_dir="${DRIVE_HOME}/selfplay/sgf" \
  --softpick_move_cutoff=20 \
  --num_readouts=200 \
  --parallel_readouts=16 \
  --num_games_share_tree=1 \
  --num_games=5 \
  2>&1 | tee "${DRIVE_HOME}/selfplay/run_selfplay.log"

#  --load_file="${DRIVE_HOME}/checkpoints/model1_epoch2.h5" \
#  --dirichlet_noise_weight=0.025 \

