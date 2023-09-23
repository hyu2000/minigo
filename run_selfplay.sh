DRIVE_HOME=/content/drive/MyDrive/dlgo
DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/9x9

export BOARD_SIZE=9

SELFPLAY_DIR="${DRIVE_HOME}/selfplay"
mkdir ${SELFPLAY_DIR}

for i in {1..1}
do
  python3 run_selfplay.py \
    --verbose=0 \
    --selfplay_dir="${SELFPLAY_DIR}/train" \
    --holdout_dir="${SELFPLAY_DIR}/val" \
    --sgf_dir="${SELFPLAY_DIR}/sgf" \
    --softpick_move_cutoff=0 \
    --dirichlet_noise_weight=0.25 \
    --num_readouts=200 \
    --full_readout_prob=1.0 \
    --reduce_symmetry_before_move=0 \
    --parallel_readouts=16 \
    --holdout_pct=0 \
    --num_games=200 \
    --load_file="${DRIVE_HOME}/checkpoints/model4_4.mlpackage" \
    --resign_threshold=-1 \
    2>&1 | tee "${SELFPLAY_DIR}/run_selfplay${i}.log" &

done
