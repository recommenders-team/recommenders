# SLi-Rec TensorFlow → PyTorch migration — design

## Goal

Port the SLi-Rec sequential recommender (currently TF-1.x graph mode) to PyTorch so
it reproduces the TF model's metrics. Concrete success gate: the quick-start
functional test `test_slirec_quickstart_functional` (10 epochs, batch 400, seed 42,
on the sampled Amazon Movies&TV data) records **AUC ≈ 0.7183** within the functional
test tolerance, using the *same* framework-agnostic metric code so numbers are
directly comparable. TF reference (from a real run): auc 0.7174, group_auc 0.7073,
logloss 0.6149, mean_mrr 0.4835, ndcg@2 0.3939, ndcg@4 0.4982, ndcg@6 0.5503;
untrained baseline auc ≈ 0.4857.

## Decisions (confirmed with the user)

- **Self-contained PyTorch stack**, mirroring the existing PyTorch `lightgcn.py` in
  the same package: a standalone `nn.Module` with `fit()` / `run_eval()` / `predict()`,
  device handling, torch `state_dict` checkpoints, reusing only the framework-agnostic
  metric functions in `deeprec_utils`. It touches **zero** shared TF files, so the 5
  other TF sequential models (asvd, caser, gru, nextitnet, sum) keep working.
- **SLi-Rec first** as a vertical slice; the reusable pieces (sequential dataset,
  time-aware cell, train/eval loop) are factored so the other 5 become cheap
  fast-follow PRs. Not bundling all 6 into one un-reviewable branch.
- **API** mirrors `lightgcn.py`'s shape (nn.Module + fit/run_eval/predict). The
  constructor takes `hparams` (sli_rec.yaml drives ~20 config values), the iterator
  class, and a seed — the same public surface the notebook uses — with the training
  logic in an internal train loop.
- **TF files kept** (untouched) so they serve as the golden reference for parity
  during development. Parity is verified in scratch scripts; committed unit tests
  assert internal consistency, shapes, and a fit smoke run (no committed TF fixtures).

## New files

- `recommenders/models/deeprec/models/sequential/pytorch/__init__.py`
- `.../pytorch/rnn_cell_pytorch.py` — `Time4LSTMCell(nn.Module)` + `time4lstm_scan`
- `recommenders/models/deeprec/io/sequential_dataset_pytorch.py` — generator mirroring
  `SequentialIterator.load_data_from_file` verbatim (8-col parse, log-scaled time
  features with 0.5 floor, left-align/most-recent truncation, in-batch negatives,
  `<5 => skip`, per-epoch shuffle only when sampling).
- `.../pytorch/sli_rec_pytorch.py` — `SLI_RECModel(nn.Module)` (forward == the TF
  `_build_seq_graph`) plus fit/run_eval/predict/load_model.
- `.../pytorch/sequential_base_pytorch.py` — shared embeddings, `_attention` (unmasked
  ASVD), `_attention_fcn` (masked), `_fcn_net` MLP head, softmax pairwise loss,
  unique-embedding regularization, per-parameter grad clip.
- `tests/unit/recommenders/models/test_sli_rec_pytorch.py` — plain-function pytest.
- `examples/00_quick_start/sequential_recsys_amazondataset.ipynb` — imports repointed
  to the PyTorch modules; every other cell kept so the functional test runs unchanged.

## Reused unchanged

`deeprec_utils` (`cal_metric`, `mrr_score`, `ndcg_score`, `dcg_score`, `hit_score`,
`prepare_hparams`, `HParams`, `load_dict`), and `config/sli_rec.yaml`.

## Numeric-parity risk register (top items)

1. In-batch dynamic negative sampling RNG cannot bit-match TF → land AUC within
   tolerance, not bit-exact; verify untrained baseline (~0.4857) first.
2. `dynamic_rnn` padding: zero outputs and freeze state past `seq_len` in the scan.
3. Softmax loss = `-group * mean(log(pos_softmax))` over the full (N,group) tensor
   (negatives set to 1.0 → log 0). Keep the leading `group` factor and N*group denom.
4. Regularize only the **unique** involved item/cate embeddings, with `tf.nn.l2_loss`'s
   0.5 factor; separate embed vs layer coefficients.
5. Time-aware cell column order `[item_emb | time_from_first_action | time_to_now]`;
   `time_now = inputs[:,-1]` (input gate), `time_last = inputs[:,-2]` (forget gate).
   Category is NOT in the LSTM input.
6. `_fcn_net` order Linear → BN(momentum 0.05, eps 1e-4) → Dropout → Activation.
7. Long-term ASVD attention is **unmasked**; short-term `_attention_fcn` **is** masked.
8. `E = item_dim + cate_dim == hidden_size == attention_size` (assert at init).
9. Index 0 is both OOV and PAD with a trainable non-zero embedding → no `padding_idx`.

## Phased plan (each independently verifiable)

0. Scaffolding + config coupling asserts.
1. `Time4LSTMCell` + scan; single-step and padded-sequence parity vs TF.
2. Dataset parity: time features, padding/truncation, eval-batch array equality.
3. Loss / metrics / regularization scalar parity.
4. Sub-block forward parity (att_fea1, att_fea2, alpha, `_fcn_net`).
5. End-to-end forward parity (weights copied TF→torch, logit atol 1e-4).
6. Training loop + optimizer + early stopping; untrained baseline check.
7. Full functional-metric match (AUC ≈ 0.7183) via the rewritten notebook.
8. Regression: existing TF deeprec tests still pass.

## Results (validated)

Component parity (weights copied TF→PyTorch, on the real slirec data):

- **Time4LSTMCell + scan**: single-step `m`/`c` diff ~1e-7; full padded-sequence
  `rnn_outputs` diff ~6e-8; padded steps exactly 0.
- **SequentialDataset**: parser, eval-batch arrays, and train-batch arrays
  (including in-batch negative sampling under a shared RNG seed) all bit-identical
  to the TF `SequentialIterator`.

End-to-end training (10 epochs, batch 400, seed 42, `embed_l2=layer_l2=0`):

| | untrained | epoch 1 | epoch 10 (valid) | **test** |
|---|---|---|---|---|
| PyTorch AUC | 0.480 | 0.506 | 0.761 | **0.7361** |
| TF AUC | 0.4857 | 0.4975 | 0.7369 | 0.7174 |

The functional test asserts `auc == pytest.approx(0.7183, rel=0.1, abs=0.05)`
(accepts `[0.6465, 0.7901]`); the PyTorch test AUC **0.7361** passes comfortably and
exceeds both the target and the TF reference. group_auc 0.722 vs TF 0.7073.

Unit tests: 12/12 pass on CPU. The five other TF sequential models are byte-identical
to staging (untouched).
