for j in 512; do # batch_size
for k in 3 5 9 17; do # vocab_size
for l in 0.00001; do # learning rate
for i in 10 20 30; do # 3 seeds
/root/.local/bin/poetry run python -m emecom_gen.zoo.attval_signaling_game.train \
    --vocab_size $k \
    --max_len 32 \
    --fix_message_length false \
    --n_epochs 20000 \
    --batch_size $j \
    --sender_cell_type gru \
    --sender_hidden_size 512 \
    --sender_embedding_dim 32 \
    --sender_layer_norm true \
    --receiver_cell_type gru \
    --receiver_hidden_size 512 \
    --receiver_embedding_dim 32 \
    --receiver_layer_norm true \
    --receiver_dropout_alpha 1e-3 \
    --baseline_type baseline-from-sender \
    --beta_scheduler_type rewo \
    --beta_rewo_communication_loss_constraint 0.3 \
    --prior_type receiver \
    --n_attributes 4 \
    --n_values 4 \
    --random_seed $i \
    --exp_id 0 \
    --sender_lr $l \
    --receiver_lr $l
done
done
done
done