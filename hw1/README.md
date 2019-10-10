### HW1

## Make Dataset
`python run_bert --do_data`

## Train
`python run_bert --do_train 
--pretrained_model_name 'bert-base-uncased' 
--max_len 256 
--epochs 6 
--batch_size 2 
--learning_rate 1e-5 
--cuda 0 `

## Predict
`python run_bert --do_test 
--pretrained_model_name 'bert-base-uncased' 
--max_len 256 
--cuda 0 
--checkpoint 3`
