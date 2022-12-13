export run_cmd="utils/parallel/run.pl"
# export spark_cmd="$AM_ROOT/kaldi_utils/parallel/hope_submit_cpu.py --queue root.zw03_training.hadoop-speech.training --job-name $USER --jumper xr-ai-speech-ttsoffline03"   # sometimes may report errors
export spark_cmd="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech/tools/speech-repo/asr/offline/am_training/kaldi_utils/parallel/hope_submit_cpu.py --queue root.zw03_training.hadoop-speech.spark --job-name $USER-simudata --mem 3G"
export train_cmd=$run_cmd
