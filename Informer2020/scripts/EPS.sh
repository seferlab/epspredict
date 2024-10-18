python -u main_informer.py \
 --model informer \
 --data EPS \
 --sector all_sector \
 --season pre-pandemic \
 --freq m \
 --features MS \
 --seq_len 4 \
 --label_len 1 \
 --pred_len 1 \
 --enc_in 57 \
 --dec_in 57 \
 --e_layers 2 \
 --d_layers 1 \
 --attn prob \
 --des 'Exp' \
 --itr 4 \
 --factor 3 &

#  python -u main_informer.py \
#  --model informer \
#  --data EPS \
#  --sector all-sector \
#  --season post-pandemic \
#  --freq m \
#  --features MS \
#  --seq_len 4 \
#  --label_len 1 \
#  --pred_len 1 \
#  --enc_in 57 \
#  --dec_in 57 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --attn prob \
#  --des 'Exp' \
#  --itr 1 \
#  --factor 3 &


#  python -u main_informer.py \
#  --model informer \
#  --data EPS \
#  --sector financials \
#  --season post-pandemic \
#  --freq m \
#  --features MS \
#  --seq_len 4 \
#  --label_len 1 \
#  --pred_len 1 \
#  --enc_in 57 \
#  --dec_in 57 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --attn prob \
#  --des 'Exp' \
#  --itr 1 \
#  --factor 3 &

#   python -u main_informer.py \
#  --model informer \
#  --data EPS \
#  --sector financials \
#  --season pre-pandemic \
#  --freq m \
#  --features MS \
#  --seq_len 4 \
#  --label_len 1 \
#  --pred_len 1 \
#  --enc_in 57 \
#  --dec_in 57 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --attn prob \
#  --des 'Exp' \
#  --itr 1 \
#  --factor 3 &


