#!/bin/bash

#  2022   jerrypeng1937@gmail.com
#  demo on the usage of plda variants

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


stage=0.0

# please modify these path variables
train_embds=exp/ecapa-tdnn/cnceleb1/train/embeds.scp
utt2spk=/lan/ibdata/jerry/research/cnceleb/xvector/data/train/utt2spk
enroll_embds=exp/ecapa-tdnn/cnceleb1/eval_enroll/spk_embeds.scp
test_embds=exp/ecapa-tdnn/cnceleb1/eval_test/embeds.scp
exp=exp/ecapa-tdnn/cnceleb1/plda
trials=cnceleb/interview
within_covar=false
between_covar=true

. ./cmd.sh
. utils/parse_options.sh


umask 000
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=8

# for score evaluation
evaluate() {
  scores=$1
  local/prepare_for_eer.py $trials $scores | compute-eer - || exit 1;
}


if (( $(echo "$stage <= 0.5" | bc -l) )); then
  log "======================================================"
  log ">> stage 0.5: train out-of-domain sparse plda model <<"
  log "======================================================"
  python local/sparse_plda.py \
    --prior_within 0.00 \
    --prior_between 0.00 \
    --utt2spk_fn $utt2spk \
    --sparse_penalty 0.001 \
    --sparse_within_covar $within_covar \
    --sparse_between_covar $between_covar \
    --embed_rspecifier "ark:ivector-normalize-length scp:$train_embds ark:- |" \
    --plda_wxfilename $exp/splda || exit 1;
fi

if (( $(echo "$stage <= 0.51" | bc -l) )); then
  log "========================="
  log ">> stage 0.51: scoring <<"
  log "========================="
  $run_cmd $exp/score/log/sparse_plda.log \
    ivector-plda-scoring --normalize-length=true \
      "ivector-copy-plda --smoothing=0.0 $exp/splda - |" \
      "ark:ivector-normalize-length scp:$enroll_embds ark:- |" \
      "ark:ivector-normalize-length scp:$test_embds ark:- |" \
      "cat '$trials' | cut -d\  --fields=1,2 |" $exp/score/splda || { log "failed to score"; exit 1; }

  evaluate $exp/score/splda
fi

if (( $(echo "$stage <= 0.7" | bc -l) )); then
  log "==========================================================="
  log ">> stage 0.7: sup-init train out-of-domain bayes plda model <<"
  log "==========================================================="
  # note that our implementation of bayes_plda does not support num_spk to be too large
  # the implementation can be improved to reduce mem usage, but we didn't spend time on it
  python local/bayes_plda.py \
    --utt2spk_fn $utt2spk \
    --embed_rspecifier "ark:ivector-normalize-length scp:$train_embds ark:- |" \
    --plda_wxfilename $exp/sbplda || exit 1;
fi

if (( $(echo "$stage <= 0.71" | bc -l) )); then
  log "========================="
  log ">> stage 0.71: scoring <<"
  log "========================="
  $run_cmd $exp/score/log/supinit_bayes_plda.log \
    ivector-plda-scoring --normalize-length=true \
      "ivector-copy-plda --smoothing=0.0 $exp/sbplda - |" \
      "ark:ivector-normalize-length scp:$enroll_embds ark:- |" \
      "ark:ivector-normalize-length scp:$test_embds ark:- |" \
      "cat '$trials' | cut -d\  --fields=1,2 |" $exp/score/sbplda || { log "failed to score"; exit 1; }

  evaluate $exp/score/sbplda
fi

log "task completed for $trials!"

log "task report: "
log "sparse plda"
  evaluate $exp/score/splda
