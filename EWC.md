data 관련 zip 폴더 Deep-Emotion 안에 압축 풀어주세요

1. train_0~train_3까지 ewc 써서 training 시키는 코드

```
python main_ewc.py  -t True -d ./data_/csv -n ac -e 100 -bs 128 -lr 0.005
```

(-d data 경로 / -n .pt name / -e epoch per task / -bs 128 / -lr  learning rate )

* 각 task 마다 완전히 수렴할때까지 돌도록 -e 크기 설정할것 

2. 모델에 이어서 트레이닝 하는 코드

```
python main2.py  -t True --data ./data_ -fol train_3 -pt /home/hwichang/emo/Deep-Emotion/pt_resume/deep_emotion-0-128-0.005.pt -lr 0.005 -e 100 -bs 128 -re 1
```

-pt .pt경로 / -re 이어서할건지 여부 / -fol data 폴더명

3. validation set에 대한 평가방법 

```
python visualize0.py -d ./eval --model /home/hwichang/emo/Deep-Emotion/pt_total/deep_emotion-100-128-0.005.pt -t True
```

--model : .pt 경로 

( pt: ewc쓴 .pt / pt_resume : ewc 안쓰고 이어서 트레이닝한 . pt / pt_total : 전체 데이터셋 한번에 돌린 .pt )

4. data folder

data : task0~ task4 합친 데이터
data_ : task0~task3 하나씩 나눠놓은 데이터
data_total : task0~task3합친데이터
