set -e
# source activate gans
#declare -a modelnames=("fren_mm" "fren_mm_te") # "fren_mm_te" "fren_mm_te_ri" "fren_mm_ns" "fren_mm_ns_te" "fren_mm_ns_te_ri")
# declare -a modelnames=("fren.transformer.vmf" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4" "fren.transformer.vmf.5")
# declare -a modelnames=("deen.transformer.vmf") # "fren.transformer.vmf.3") #("fren.transformer.vmf.long" "fren.transformer.vmf.5" "fren.transformer.vmf.6" "fren.transformer.vmf.7") # "fren.transformer.vmf.1" "fren.transformer.vmf.2" "fren.transformer.vmf.3" "fren.transformer.vmf.4")
# declare -a modelnames=("deen_transformer")
BS=1
modelname=deen.transformer.vmf.norm.dot
echo $modelname
# for i in 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000; do
    # -model ~/models/deen_transformer_sm_model_step_70000.pt \
for i in 0; do
export CUDA_VISIBLE_DEVICES=1
python -u translate.py\
    -decode_loss dot\
    -gpu 0\
    -model logs/deen.transformer.vmf.norm.dot/model_step_16000.pt \
    -src ../.data/deen/tst201516.tok.true.de\
    -output logs/$modelname/step_${i}_pred.dev.bs$BS.en\
    -batch_size 4000\
    -batch_type tokens\
    -beam_size $BS\
    -replace_unk | tee logs/decodelogs_deen.out 2>&1
./evaluate.sh logs/$modelname/step_${i}_pred.dev.bs$BS.en ../.data/deen/tst201516.en en >> logs/$modelname/devbleus.txt
# ./evaluate.sh logs/$modelname/step_${i}_pred.dev.bs$BS.en ../../kumarvon2018-data/deen/tst201314.en en >> logs/$modelname/devbleus.txt
done

exit 0
bs1  25.64
bs2  25.59
bs3  25.71
bs4  25.72
bs5  25.57
bs10 25.54
bs20 25.44
