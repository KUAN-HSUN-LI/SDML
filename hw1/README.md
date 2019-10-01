### HW1

## Make Dataset
give each oov an embedding:
`python3 preprocess.py`

treat all oov as unk:
`python3 preprocess.py --oov_as_unk`

## Train
--save_dir_name: model(default)

--cuda ordinal: 0(default)

`python3 train.py --save_dir_name 'model_L2Reg_unkFalse' --cuda 0`

## Predict
`python3 predict.py`
