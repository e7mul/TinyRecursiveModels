run_name="mdm_experiment"
torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm_mdm \
data_paths="[data/mdm_dataset/]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=0 weight_decay=1.0 puzzle_emb_weight_decay=0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True