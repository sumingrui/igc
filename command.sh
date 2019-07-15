# 脚本控制测试案例

# 测试1：决定dropout数值
# 普通igc卷积 
for i in $(seq 1 5)
do 

python s-igc.py --network igc1 --depth 5 --firstgroup 6 --secondgroup 2 --dataset indian_pines --spatial-size 7 --bf 0 --train-scale 0.3 --batch-size 32 \
--num-epochs 20 --lr 0.001 --lr-factor 0.333 --lr-callback 1 --lr-byepoch 30 --optimizer adam --dropout 0 --pooling max

done

sleep 5;

# 修改dropout 0.1
for i in $(seq 1 5)
do 

python s-igc.py --network igc1 --depth 5 --firstgroup 4 --secondgroup 3 --dataset indian_pines --spatial-size 7 --bf 0 --train-scale 0.3 --batch-size 32 \
--num-epochs 200 --lr 0.001 --lr-factor 0.333 --lr-callback 1 --lr-byepoch 30 --optimizer adam --dropout 0.1 --pooling max

done

# 普通plain卷积
# python s-igc.py --network plain --depth 5 --firstgroup 1 --secondgroup 12 --dataset indian_pines --spatial-size 7 --bf 0 --train-scale 0.3 --batch-size 32 \
# --num-epochs 20 --lr 0.001 --lr-factor 0.333 --lr-callback 1 --lr-byepoch 30 --optimizer adam --dropout 0 --pooling max