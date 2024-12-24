nice -n 19 python run.py train delaney --task reg --device cuda:0 --batch_size 32 --train_epoch 100 --lr 0.0005 --fold 3 --dropout 0.05 --attn_head 6 --attn_layers 3 --output_dim 128 --D 8 --metric rmse

