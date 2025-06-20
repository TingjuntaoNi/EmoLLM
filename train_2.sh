BACKBONE=roberta-base
PROMPTLEN=100

for SENTIMENT in 'desire' 'disgust' 'disapproval' 'anger' \
                 'annoyance' 'confusion' 'fear' 'disappointment' 'grief' 'sadness' \
                 'nervousness' 'embarrassment' 'remorse'
do
  for seed in 1 3 5 7 9 11 13 15 17 19 42 100
  do
    echo "Training sentiment=$SENTIMENT, seed=$seed"

    CUDA_VISIBLE_DEVICES=0 \
    python train.py \
      --output_dir outputs/${SENTIMENT}/seed_${seed} \
      --backbone $BACKBONE \
      --prompt_len $PROMPTLEN \
      --sentiment=$SENTIMENT \
      --seed $seed \
      --num_train_epochs 100

    echo "✅ Done: $SENTIMENT - seed $seed"
  done
done