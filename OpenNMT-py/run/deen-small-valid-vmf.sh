source activate vivo
#current hyperparameters have performed the best,current-300-best model is in vmf.4
SAVEDIR='logs/deen.transformer.vmf.valid'
export CUDA_VISIBLE_DEVICES=3
mkdir -p $SAVEDIR
python -u train.py\
    -data ~/data/deen/conmt300/data\
    -train_from /home/ubuntu/models/deen_transformer_vmf_model_step_38000.pt\
    -train_steps 0 \
    -valid_steps 1 \
    -topk_acc 10\
    -generator_function continuous-linear\
    -loss nllvmf\
    -generator_layer_norm\
    -lambda_vmf 0.2\
    -center\
    -share_decoder_embeddings > $SAVEDIR/log.out 2>&1 




# -train_from /home/ubuntu/artidoro-seq2seq-con/OpenNMT-py/logs/deen.transformer.share.fix/model_step_70000.pt\