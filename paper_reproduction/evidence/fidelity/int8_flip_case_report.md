# Int8 Flip Case Report

## CCNN-int8

- Flips: 67; stable trials: 1733.
- Harmful: 25; beneficial: 18; both-wrong different class: 24.
- Mean host margin, flips: 0.6666; stable: 14.2952.
- Mean logit MAE, flips: 0.6068; stable: 0.6463.
- Mean cosine similarity, flips: 0.993959; stable: 0.997136.

## EEGNet-int8

- Flips: 34; stable trials: 1766.
- Harmful: 11; beneficial: 6; both-wrong different class: 17.
- Mean host margin, flips: 0.0333; stable: 0.5394.
- Mean logit MAE, flips: 0.0212; stable: 0.0177.
- Mean cosine similarity, flips: 0.999387; stable: 0.999776.

## FBtCNN-int8

- Flips: 6; stable trials: 1794.
- Harmful: 2; beneficial: 2; both-wrong different class: 2.
- Mean host margin, flips: 0.1002; stable: 8.2595.
- Mean logit MAE, flips: 0.0540; stable: 0.0485.
- Mean cosine similarity, flips: 0.999768; stable: 0.999834.

Interpretation: these are output-level case classifications. They identify whether a flip changes correctness, but not the internal layer where the decision diverged.
