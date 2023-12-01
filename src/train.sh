python3 train.py --device 0 \
                 --n_epochs 10   \
                 --batch_size 8 \
                 --learning_rate 0.1   \
                 --accumulation_steps 1 \
                 --evaluation_steps 16 \
                 --model_save_dir ../model/example \
                 --model_config ../model/example/config.json \