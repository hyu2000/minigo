DRIVE_HOME=/content/drive/MyDrive/dlgo
DRIVE_HOME=/Users/hyu/PycharmProjects/dlgo/5x5

export BOARD_SIZE=5

SELFPLAY_DIR="${DRIVE_HOME}/selfplay"
mkdir ${SELFPLAY_DIR}

for i in {1..4}
do
  python run_selfplay.py \
    --verbose=0 \
    --selfplay_dir="${SELFPLAY_DIR}/train" \
    --holdout_dir="${SELFPLAY_DIR}/val" \
    --sgf_dir="${SELFPLAY_DIR}/sgf" \
    --softpick_move_cutoff=6 \
    --num_readouts=200 \
    --full_readout_prob=1.0 \
    --reduce_symmetry=True \
    --parallel_readouts=16 \
    --holdout_pct=0 \
    --dirichlet_noise_weight=0.25 \
    --num_games=20 \
    --load_file="${DRIVE_HOME}/checkpoints/model13_epoch2.h5" \
    --resign_threshold=-1 \
    2>&1 | tee "${SELFPLAY_DIR}/run_selfplay${i}.log" &

done
