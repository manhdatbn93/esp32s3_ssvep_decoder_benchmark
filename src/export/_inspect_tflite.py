"""Quick TFLite model inspector."""
import os, sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "export")
for d in sorted(os.listdir(base)):
    dp = os.path.join(base, d)
    if not os.path.isdir(dp):
        continue
    for f in sorted(os.listdir(dp)):
        if not f.endswith(".tflite"):
            continue
        fp = os.path.join(dp, f)
        try:
            interp = tf.lite.Interpreter(model_path=fp)
            interp.allocate_tensors()
            inp = interp.get_input_details()[0]
            out = interp.get_output_details()[0]
            ops = interp._get_ops_details()
            op_names = sorted(set(o["op_name"] for o in ops))
            size_kb = os.path.getsize(fp) / 1024
            print(f"{d}/{f} ({size_kb:.1f} KB)")
            print(f"  input:  shape={list(inp['shape'])} dtype={inp['dtype'].__name__}")
            print(f"  output: shape={list(out['shape'])} dtype={out['dtype'].__name__}")
            print(f"  ops: {op_names}")
        except Exception as e:
            print(f"{d}/{f}: ERROR {e}")
