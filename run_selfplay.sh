DRIVE_HOME=/content/drive/MyDrive/dlgo
DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/9x9

export BOARD_SIZE=9

SELFPLAY_DIR="${DRIVE_HOME}/selfplay"
mkdir ${SELFPLAY_DIR}

for i in {1..5}
do
  python3 run_selfplay.py \
    --verbose=0 \
    --selfplay_dir="${SELFPLAY_DIR}/train" \
    --holdout_dir="${SELFPLAY_DIR}/val" \
    --sgf_dir="${SELFPLAY_DIR}/sgf" \
    --softpick_move_cutoff=6 \
    --num_readouts=800 \
    --num_fast_readouts=100 \
    --full_readout_prob=0.5 \
    --parallel_readouts=16 \
    --holdout_pct=0 \
    --num_games_share_tree=1 \
    --dirichlet_noise_weight=0.25 \
    --num_games=1000 \
    --load_file="${DRIVE_HOME}/pbt/model17_epoch6.h5" \
    --resign_threshold=-0.95 \
    2>&1 | tee "${SELFPLAY_DIR}/run_selfplay${i}.log" &

done
