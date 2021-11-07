DRIVE_HOME=/content/drive/MyDrive/dlgo
DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/9x9

export BOARD_SIZE=9

#mkdir ${DRIVE_HOME}/selfplay

for i in {1..2}
do
  python3 run_selfplay.py \
    --verbose=0 \
    --selfplay_dir="${DRIVE_HOME}/selfplay/train" \
    --holdout_dir="${DRIVE_HOME}/selfplay/val" \
    --sgf_dir="${DRIVE_HOME}/selfplay/sgf" \
    --softpick_move_cutoff=8 \
    --num_readouts=400 \
    --parallel_readouts=16 \
    --holdout_pct=0 \
    --num_games_share_tree=1 \
    --dirichlet_noise_weight=0.05 \
    --load_file="${DRIVE_HOME}/checkpoints/model10_epoch4.h5" \
    --num_games=500 \
    2>&1 | tee "${DRIVE_HOME}/selfplay/run_selfplay${i}.log" &

done
