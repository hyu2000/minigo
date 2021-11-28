DRIVE_HOME=/content/drive/MyDrive/dlgo
DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/9x9

export BOARD_SIZE=9

SELFPLAY_DIR="${DRIVE_HOME}/selfplay"
mkdir ${SELFPLAY_DIR}

for i in {1..3}
do
  python3 run_endgame.py \
    --verbose=0 \
    --tar_dir="${DRIVE_HOME}/games" \
    --selfplay_dir="${SELFPLAY_DIR}/train" \
    --holdout_dir="${SELFPLAY_DIR}/val" \
    --sgf_dir="${SELFPLAY_DIR}/sgf" \
    --softpick_move_cutoff=6 \
    --num_readouts=600 \
    --num_fast_readouts=100 \
    --full_readout_prob=1. \
    --parallel_readouts=16 \
    --holdout_pct=0 \
    --dirichlet_noise_weight=0.25 \
    --num_games=2 \
    --load_file="${DRIVE_HOME}/pbt/model16_epoch1.h5" \
    --resign_threshold=-1 \
    2>&1 | tee "${SELFPLAY_DIR}/run_selfplay${i}.log" &

done
