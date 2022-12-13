# Implementation of Sparse PLDA and Bayes PLDA

## Core files

`plda.py`: python version of kaldi's two-covariance PLDA

`bayes_plda.py`: Implementation of paper: **Unsupervised Bayesian Adaptation of PLDA for Speaker Verification**

`sparse_plda.py`: Implementation of paper: **Covariance Regularization of Probabilistic Linear Dscriminant Analysis**


The i/o follows kaldi's style.

Usage example:

```
  python sparse_plda.py \
    --utt2spk_fn $path_to_utt2spk \
    --embed_rspecifier "ark:ivector-normalize-length scp:$train_embeds_scp ark:-|" \
    --plda_wxfilename $plda_mdl
``` 
    
After training, the plda model is stored in path: $plda_mdl, which can be used by kaldi's `ivector-plda-scoring`, e.g.,
```
ivector-plda-scoring --normalize-length=true \
  "ivector-copy-plda --smoothing=0.0 $plda_mdl - |" \
  "ark:ivector-normalize-length scp:$enroll_embeds_scp ark:- |" \
  "ark:ivector-normalize-length scp:$test_embeds_scp ark:- |" \
  "cat '$trials' | cut -d\  --fields=1,2 |" $scores
```

If you are familiar with kaldi recipes, see demo.sh
  
