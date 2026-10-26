# TFLite Operator And Quantization Audit

TensorFlow host inspection was used with `experimental_preserve_all_tensors=True` when available.

## Backend Operator Summary

### CCNN-f32

- Representative file: `<DEV_REPO>\export\strict_loso_inner_val\12jfpm_ccnn_fold01_test_s1\ccnn_float32.tflite`
- Input dtype(s): <class 'numpy.float32'>
- Output dtype(s): <class 'numpy.float32'>
- Input quantization: `[[0.0, 0]]`
- Output quantization: `[[0.0, 0]]`
- Unique ops: CONV_2D, FULLY_CONNECTED, RESHAPE, TRANSPOSE
- Host TensorFlow delegate entry observed: True
- Custom/unsupported ops observed by host interpreter: none observed

### CCNN-int8

- Representative file: `<DEV_REPO>\export\strict_loso_inner_val\12jfpm_ccnn_fold01_test_s1_int8\ccnn_int8.tflite`
- Input dtype(s): <class 'numpy.float32'>
- Output dtype(s): <class 'numpy.float32'>
- Input quantization: `[[0.0, 0]]`
- Output quantization: `[[0.0, 0]]`
- Unique ops: CONV_2D, DEQUANTIZE, FULLY_CONNECTED, QUANTIZE, RESHAPE, TRANSPOSE
- Host TensorFlow delegate entry observed: True
- Custom/unsupported ops observed by host interpreter: none observed

### ConvCA-f32

- Representative file: `<DEV_REPO>\export\strict_loso_inner_val\12jfpm_convca_fold01_test_s1\convca_float32.tflite`
- Input dtype(s): <class 'numpy.float32'>
- Output dtype(s): <class 'numpy.float32'>
- Input quantization: `[[0.0, 0]]`
- Output quantization: `[[0.0, 0]]`
- Unique ops: BATCH_MATMUL, CONV_2D, DIV, FULLY_CONNECTED, MUL, RESHAPE, SQRT, SUM
- Host TensorFlow delegate entry observed: True
- Custom/unsupported ops observed by host interpreter: none observed

### EEGNet-f32

- Representative file: `<DEV_REPO>\export\strict_loso_inner_val\12jfpm_eegnet_fold01_test_s1\eegnet_float32.tflite`
- Input dtype(s): <class 'numpy.float32'>
- Output dtype(s): <class 'numpy.float32'>
- Input quantization: `[[0.0, 0]]`
- Output quantization: `[[0.0, 0]]`
- Unique ops: AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, ELU, FULLY_CONNECTED, MEAN, PAD, RESHAPE, STRIDED_SLICE
- Host TensorFlow delegate entry observed: True
- Custom/unsupported ops observed by host interpreter: none observed

### EEGNet-int8

- Representative file: `<DEV_REPO>\export\strict_loso_inner_val\12jfpm_eegnet_fold01_test_s1_int8\eegnet_int8.tflite`
- Input dtype(s): <class 'numpy.float32'>
- Output dtype(s): <class 'numpy.float32'>
- Input quantization: `[[0.0, 0]]`
- Output quantization: `[[0.0, 0]]`
- Unique ops: AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, DEQUANTIZE, ELU, FULLY_CONNECTED, MEAN, PAD, QUANTIZE, RESHAPE, STRIDED_SLICE
- Host TensorFlow delegate entry observed: True
- Custom/unsupported ops observed by host interpreter: none observed

### FBtCNN-f32

- Representative file: `<DEV_REPO>\export\strict_loso_inner_val\12jfpm_fbtcnn_fold01_test_s1\fbtcnn_float32.tflite`
- Input dtype(s): <class 'numpy.float32'>
- Output dtype(s): <class 'numpy.float32'>
- Input quantization: `[[0.0, 0]]`
- Output quantization: `[[0.0, 0]]`
- Unique ops: CONV_2D, ELU, FULLY_CONNECTED, PAD, RESHAPE, SUM, TRANSPOSE
- Host TensorFlow delegate entry observed: True
- Custom/unsupported ops observed by host interpreter: none observed

### FBtCNN-int8

- Representative file: `<DEV_REPO>\export\strict_loso_inner_val\12jfpm_fbtcnn_fold01_test_s1_int8\fbtcnn_int8.tflite`
- Input dtype(s): <class 'numpy.float32'>
- Output dtype(s): <class 'numpy.float32'>
- Input quantization: `[[0.0, 0]]`
- Output quantization: `[[0.0, 0]]`
- Unique ops: CONV_2D, DEQUANTIZE, ELU, FULLY_CONNECTED, PAD, QUANTIZE, RESHAPE, SUM, TRANSPOSE
- Host TensorFlow delegate entry observed: True
- Custom/unsupported ops observed by host interpreter: none observed

## Inspection Notes

- CCNN-f32: ok
- CCNN-int8: ok
- ConvCA-f32: ok
- EEGNet-f32: ok
- EEGNet-int8: ok
- FBtCNN-f32: ok
- FBtCNN-int8: ok

Kernel selection note: the current host TFLite inspection lists graph operators and tensor quantization metadata. TensorFlow host `DELEGATE` entries are not TFLite Micro graph operators and are excluded from the unique-op lists above. This audit does not prove which optimized or reference TFLM kernel was selected on ESP32-S3 for each invocation.
