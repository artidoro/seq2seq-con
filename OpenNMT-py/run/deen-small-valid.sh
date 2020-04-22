source activate vivo
#current hyperparameters have performed the best,current-300-best model is in vmf.4
SAVEDIR='logs/deen.transformer.valid'
export CUDA_VISIBLE_DEVICES=7
mkdir -p $SAVEDIR
python -u train.py\
    -data ~/data/deen/conmt300/data\
    -train_from /home/ubuntu/models/deen_transformer_sm_model_step_24000.pt\
    -train_steps 0 \
    -valid_steps 1 \
    -topk_acc 10\
    -gpu_ranks 0 > $SAVEDIR/log.out 2>&1 


# /home/ubuntu/models/deen_transformer_vmf_model_step_38000.pt\
#/home/ubuntu/artidoro-seq2seq-con/OpenNMT-py/logs/deen.transformer.cos.share.fix.3/model_step_100000.pt\