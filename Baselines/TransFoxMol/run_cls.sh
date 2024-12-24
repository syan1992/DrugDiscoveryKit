nice -n 19 python run.py train tox21 --task clas --device cuda:0 --batch_size 32 --train_epoch 50 --lr 0.0005 --fold 3 --dropout 0.05 --attn_head 6 --attn_layers 2 --output_dim 256 --D 4 

