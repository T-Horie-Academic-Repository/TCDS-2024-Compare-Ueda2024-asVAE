for j in 1024; do # batch_size
for k in 3 5 9 17; do # vocab_size
for l in 1e-5; do # learning rate
for i in {0..9}; do # 3 seeds
if [ $k -eq 3 ]; then
    iter=3000
else
    iter=1500
fi
/root/.local/bin/poetry run python -m emecom_gen.zoo.attval_signaling_game.train \
    --vocab_size $k \
    --max_len 32 \
    --fix_message_length false \
    --n_epochs $iter \
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
    --exp_id $i \
    --sender_lr $l \
    --receiver_lr $l
done
done
done
done
