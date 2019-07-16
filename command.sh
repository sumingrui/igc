# 脚本控制测试案例

:<<!
测试1：决定dropout数值
测试2：决定组卷积分组策略
测试3：maxpooling还是avgpooling还是用strides下采样

!
# for i in $(seq 1 5)
# do 
# python s-igc.py --network igc1 --depth 5 --firstgroup 6 --secondgroup 2 --dataset indian_pines --spatial-size 7 --bf 0 --train-scale 0.3 --batch-size 32 \
# --num-epochs 130 --lr 0.001 --lr-factor 0.333 --lr-callback 1 --lr-byepoch 30 --optimizer adam --dropout 0 --pooling avg
# done

# sleep 5;


# 测试titan xp下面igc的速度
for i in $(seq 1 5)
do
python s-igc.py --network igc --depth 5 --firstgroup 6 --secondgroup 2 --dataset indian_pines --spatial-size 7 --bf 0 --train-scale 0.3 --batch-size 32 \
--num-epochs 130 --lr 0.001 --lr-factor 0.333 --lr-callback 1 --lr-byepoch 30 --optimizer adam
done

# sleep 5;

# 测试titan xp下plain速度
for i in $(seq 1 5)
do
python s-igc.py --network plain --depth 5 --firstgroup 1 --secondgroup 12 --dataset indian_pines --spatial-size 7 --bf 0 --train-scale 0.3 --batch-size 32 \
--num-epochs 130 --lr 0.001 --lr-factor 0.333 --lr-callback 1 --lr-byepoch 30 --optimizer adam
done

# 测试resnet
python s-igc.py --network resnet --depth 5 --firstgroup 1 --secondgroup 12 --dataset indian_pines --spatial-size 7 --bf 0 --train-scale 0.3 --batch-size 32 \
--num-epochs 130 --lr 0.001 --lr-factor 0.333 --lr-callback 1 --lr-byepoch 30 --optimizer adam