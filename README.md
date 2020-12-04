# self-explaining-NLP
Code, models and Datasets for[《Self-Explaining Structures Improve NLP Models》](http://arxiv.org/abs/2012.01786).

## installation 
`pip install -r requirements.txt`

## Prepare Datasets and Models
- Download the SST-5 dataset, the official corpus can be found [HERE](https://nlp.stanford.edu/sentiment/index.html).
We provide processed raw text which you can download [HERE](https://drive.google.com/drive/folders/1TYR-yRw3NXqfXnMSvFDxGTdf1urGfrPY?usp=sharing).
Save the processed raw text dataset at `[SST_PATA_PATH]`.
- Download the SNLI dataset, the official corpus can be found [HERE](https://nlp.stanford.edu/projects/snli/).
Save the SNLI dataset at `[SNLI_PATA_PATH]`.
- Download the vanilla RoBERTa-base model released by HuggingFace. Save the model at `[ROBERTA_BASE_PATH]`,
it can be found [HERE](https://huggingface.co/roberta-base)
- Download the model checkpoints we trained for different tasks. You can use our checkpoint for evaluation.
the checkpoints can be download [HERE](https://drive.google.com/drive/folders/1RV5OJSzN_7p-YkjkmAhq2vzhouZEtzSS?usp=sharing)

## Reproduce paper results step by step
In this paper, we utilize self-explaining structures in different NLP tasks. This repo contains all train 
and evaluate codes, but here, we only provide commands for SST-5 task as an example. 
For other tasks, you can reproduce the results simply by modifying the commands.

### 1.Train the self-explaining model
SST-5 is a task with five classes, so we should modify the Roberta-base config file.
Open `[ROBERTA_BASE_PATH]\config.json` and set `num_labels=5`. Then run the following commands.
```bash
cd explain
python trainer.py \
--bert_path [ROBERTA_BASE_PATH] \
--data_dir [SST_PATA_PATH] \
--task sst5 \
--save_path [SELF_EXPLAINING_MODEL_CHECKPOINTS] \
--gpus=0,1,2,3  \
--precision 16 \
--lr=2e-5 \
--batch_size=10 \
--lamb=1.0 \
--workers=4 \
--max_epoch=20
```
After training, the checkpoints and training log will be saved at `[SELF_EXPLAINING_MODEL_CHECKPOINTS]`.
### 2.Evaluate the self-explaining model
Run the following evaluation command to get the performance on test dataset.
You can use the checkpoint you trained or just download our checkpoint to evaluate test dataset.
After evaluation, you will get two output file at `[SPAN_SAVE_PATH]`: `output.txt` and `test.txt`.
`output.txt` records visual extract spans and prediction results.
`text.txt` only records top-ranked span as span-base test data for next stage.
```bash
cd explain
python trainer.py \
--bert_path [ROBERTA_BASE_PATH] \
--data_dir [SST_PATA_PATH] \
--task sst5 \
--checkpoint_path [SELF_EXPLAINING_MODEL_CHECKPOINTS]/***.ckpt \
--save_path [SPAN_SAVE_PATH] \
--gpus=0, \
--mode eval
```

### 3.Check the extracted span
In previous stage, we got span-based test data. You can use the same method to get span-based train data.  
To check the extracted span, we set four experiments which are full-full mode, full-span mode, span-full 
mode and span-span mode. For example, full-span mode means we use origin SST-5 train data as train data,
and use span-based test data as test data.   
You should save the origin SST-5 train data and span-base test data at `[FULL_SPAN_PATH]`
```bash
scp [SST_PATA_PATH]/train.txt  [FULL_SPAN_PATH]
scp [SPAN_SAVE_PATH]/test/txt [FULL_SPAN_PATH]
cd check
python trainer.py \
--bert_path [ROBERTA_BASE_PATH] \
--data_dir [FULL_SPAN_PATH] \
--task sst5 \
--save_path [CHECK_MODEL_CHECKPOINTS] \
--gpus=0,1,2,3  \
--precision 16 \
--lr=2e-5 \
--batch_size=10 \
--workers=4 \
--max_epoch=20
```
