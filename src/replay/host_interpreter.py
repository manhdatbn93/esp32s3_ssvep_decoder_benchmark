"""TFLite interpreter configuration for exported-host references.

The retained host references were produced with TensorFlow 2.21.0 ``tf.lite.Interpreter``
and its default op resolver (``OpResolverType.AUTO``), which applies the XNNPACK delegate
automatically. ``BUILTIN_WITHOUT_DEFAULT_DELEGATES`` selects the built-in kernels without
XNNPACK. Passing ``experimental_delegates=[]`` does not disable the default delegate.
"""
from __future__ import annotations

from pathlib import Path

HOST_REFERENCE_OP_RESOLVER = "AUTO"
NO_DEFAULT_DELEGATES_OP_RESOLVER = "BUILTIN_WITHOUT_DEFAULT_DELEGATES"


def make_interpreter(model_path: str | Path, op_resolver: str = HOST_REFERENCE_OP_RESOLVER):
    import tensorflow as tf

    resolver = getattr(tf.lite.experimental.OpResolverType, op_resolver)
    return tf.lite.Interpreter(model_path=str(model_path), experimental_op_resolver_type=resolver)
