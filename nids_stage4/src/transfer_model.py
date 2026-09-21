"""
transfer_model.py
==================
Stage 4 - Component 2: Transfer Learning Mechanism

Tree ensembles don't have "layers to freeze" the way neural networks do, so
transfer learning here means something concrete and specific for each
model family:

1. TransferRandomForest — uses sklearn's `warm_start=True`. We first fit
   a forest on the SOURCE dataset (e.g. CIC-IDS2017). To fine-tune, we
   increase `n_estimators` and re-fit on the (small) TARGET dataset sample
   — with `warm_start=True`, sklearn KEEPS the existing source-trained
   trees and only grows the newly-added trees using the target data. The
   final forest is therefore a genuine mixture of source-domain and
   target-domain trees, voting together — source knowledge is retained,
   not overwritten.

2. TransferXGBoost — uses XGBoost's native continued-training support
   (`xgb_model=<previous_booster>` in `.fit()`), which resumes boosting
   from the source-trained booster's state and adds new boosting rounds
   fit on the target data. This is XGBoost's standard, documented
   mechanism for incremental/continued training and is a legitimate
   "fine-tuning" operation for gradient-boosted trees.

Both expose the same 3-way interface used in train_stage4.py:
    - fit_source(X_source, y_source)      : pretrain on the source domain
    - fine_tune(X_target_small, y_target_small) : adapt using a SMALL target sample
    - predict(X) / predict_proba(X)        : inference with the CURRENT
      (possibly fine-tuned) model state
"""

from sklearn.ensemble import RandomForestClassifier


class TransferRandomForest:
    def __init__(self, n_estimators_source: int = 100, n_estimators_finetune: int = 30,
                 random_state: int = 42):
        self.n_estimators_source = n_estimators_source
        self.n_estimators_finetune = n_estimators_finetune
        self.random_state = random_state
        self.model_ = None
        self.is_fine_tuned_ = False

    def fit_source(self, X_source, y_source):
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators_source,
            warm_start=True,          # required for incremental tree addition later
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_source, y_source)
        return self

    def fine_tune(self, X_target_small, y_target_small):
        if self.model_ is None:
            raise RuntimeError("Call fit_source() before fine_tune().")
        # Grow the forest: existing trees (trained on source data) are kept
        # as-is; the newly added n_estimators_finetune trees are trained
        # on the target-domain sample passed to THIS fit() call.
        self.model_.n_estimators += self.n_estimators_finetune
        self.model_.fit(X_target_small, y_target_small)
        self.is_fine_tuned_ = True
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class TransferXGBoost:
    """
    Requires the real `xgboost` package (falls back to a no-op-transfer
    warning if unavailable, since sklearn's HistGradientBoostingClassifier
    has no native continued-training API for this use case).
    """
    def __init__(self, n_estimators_source: int = 200, n_estimators_finetune: int = 50,
                 random_state: int = 42):
        self.n_estimators_source = n_estimators_source
        self.n_estimators_finetune = n_estimators_finetune
        self.random_state = random_state
        self.model_ = None
        self.is_fine_tuned_ = False
        self.available_ = self._check_xgboost()

    @staticmethod
    def _check_xgboost():
        try:
            import xgboost  # noqa: F401
            return True
        except ImportError:
            return False

    def fit_source(self, X_source, y_source):
        if not self.available_:
            raise RuntimeError(
                "TransferXGBoost requires the real `xgboost` package (not installed here — "
                "no network access to pip install in this dev environment). "
                "Use TransferRandomForest instead, or install xgboost on your machine."
            )
        import xgboost as xgb
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators_source, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=self.random_state, n_jobs=-1,
        )
        self.model_.fit(X_source, y_source)
        return self

    def fine_tune(self, X_target_small, y_target_small):
        import xgboost as xgb
        booster = self.model_.get_booster()
        fine_tuned = xgb.XGBClassifier(
            n_estimators=self.n_estimators_finetune, max_depth=6, learning_rate=0.05,
            eval_metric="logloss", random_state=self.random_state, n_jobs=-1,
        )
        # Resume boosting from the source-trained booster's state
        fine_tuned.fit(X_target_small, y_target_small, xgb_model=booster)
        self.model_ = fine_tuned
        self.is_fine_tuned_ = True
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
