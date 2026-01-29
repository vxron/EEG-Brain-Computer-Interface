# HADEEL

# How to package ur training so that it fits with current arch! 
# (Instructions from me and mostly chatgpt :,)<3 )

# (i.e.) EXPORT REQUIREMENTS:
# obv can have as many inner functions/implementations as needed
# ftr extraction handlers should go in utils
# ------------------------------------------------
# REQUIRED PUBLIC FUNCTIONS (EXPORT THESE)
# ------------------------------------------------
#
# 1) train_svm_on_split(...)
#    Used during frequency-pair cross-validation.
#
#    Signature concept:
#      train_svm_on_split(
#          X_train, y_train,
#          X_val, y_val,
#          seed: int,
#          **optional_hyperparams
#      ) -> (val_balanced_acc: float, val_acc: float)
#
#    Notes:
#    - X_* shape: (N, C, T), already normalized
#    - y_* labels are binary {0,1}
#    - Must fit on train split and evaluate on val split
#    - Use utils.balanced_accuracy(...) for scoring
#
#
# 2) train_svm(...)
#    Used once for FINAL training on the selected best frequency pair.
#
#    Signature concept:
#      train_svm(
#          X, y_cls,
#          seed: int,
#          val_ratio: float,
#          **optional_hyperparams
#      ) -> (model, best_state, history)
#
#    Notes:
#    - Return 3 objects to match CNN trainer API
#    - best_state and history may be minimal but MUST exist
#    - model object must be compatible with export_svm_onnx
#
#
# 3) export_svm_onnx(...)
#    Writes ONNX model used by C++ inference.
#
#    Signature concept:
#      export_svm_onnx(
#          *,
#          model,
#          n_ch: int,
#          n_time: int,
#          out_path: Path
#      ) -> Path
#
#    Notes:
#    - ONNX input name MUST be ["x"]
#    - Output name MUST be ["logits"]
#    - Output logits shape MUST be (B, 2)
#    - Input can be flattened (B, C*T) OR reshaped internally
#      but must stay consistent with C++ inference pipeline
#
#
# ------------------------------------------------
# SHARED REQUIREMENTS
# ------------------------------------------------
# - Import: numpy as np
# - Import: utils.utils as utils
# - Deterministic behavior when seed is provided
# - Handle small datasets gracefully (no crashes)
#
# ------------------------------------------------
# REQUIRED EXPORTS (top-level names)
# ------------------------------------------------
# train_svm
# train_svm_on_split
# export_svm_onnx
#
# ================================================================

import utils.utils as utils

