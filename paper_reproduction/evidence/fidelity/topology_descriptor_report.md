# Topology Descriptor Report

Scope: strict LOSO neural replay backends on ESP32-S3 DevKitC-1 v1.1.

Important: accumulation-risk labels are topology-based hypotheses, not causal layer evidence.

## CCNN-f32

- Model: `src/models/ccnn.py` / `CCNN`.
- Input: [B,512,8] PyTorch; [1,8,220,1] exported TFLite after external Goertzel FFT.
- Preprocessing: Replay crops + channel-zscores; CCNN then uses Goertzel/FFT spectrum real+imag features.
- Topology: FFT features -> spatial Conv2D -> BN -> ReLU -> Dropout -> frequency Conv2D -> BN -> ReLU -> Dropout -> Flatten -> Linear
- Operators: CONV_2D, FULLY_CONNECTED, RESHAPE, TRANSPOSE.
- Parameters: 43308; artifact size: 171.4 KiB.
- Accumulation-heavy components: Goertzel/FFT over 512 samples plus zero padding to NFFT=874; conv reductions; dense classifier
- Risk classification: high (External spectral preprocessing plus dense classifier create many reductions; high observed int8 drift makes this a priority hypothesis, not proof.)

## CCNN-int8

- Model: `src/models/ccnn.py` / `CCNN`.
- Input: [B,512,8] PyTorch; [1,8,220,1] exported TFLite after external Goertzel FFT.
- Preprocessing: Replay crops + channel-zscores; CCNN then uses Goertzel/FFT spectrum real+imag features.
- Topology: FFT features -> spatial Conv2D -> BN -> ReLU -> Dropout -> frequency Conv2D -> BN -> ReLU -> Dropout -> Flatten -> Linear
- Operators: CONV_2D, DEQUANTIZE, FULLY_CONNECTED, QUANTIZE, RESHAPE, TRANSPOSE.
- Parameters: 43308; artifact size: 46.6 KiB.
- Accumulation-heavy components: Goertzel/FFT over 512 samples plus zero padding to NFFT=874; conv reductions; dense classifier
- Risk classification: high (External spectral preprocessing plus dense classifier create many reductions; high observed int8 drift makes this a priority hypothesis, not proof.)

## ConvCA-f32

- Model: `src/models/convca.py` / `ConvCA`.
- Input: [B,512,8] trial plus [12,512,8] templates in PyTorch; exported TFLite has templates baked and accepts [1,512,8].
- Preprocessing: Replay crops + channel-zscores; class templates baked into deployed model.
- Topology: signal conv stack + template conv stack -> Pearson-like correlation over time/classes -> Linear
- Operators: BATCH_MATMUL, CONV_2D, DIV, FULLY_CONNECTED, MUL, RESHAPE, SQRT, SUM.
- Parameters: 4743; artifact size: 34.3 KiB.
- Accumulation-heavy components: correlation numerator and denominator reductions over 512 time points for 12 classes
- Risk classification: high (Correlation reductions are numerically sensitive in principle, but only f32 replay is realized here and had 100% agreement.)

## EEGNet-f32

- Model: `src/models/eegnet.py` / `EEGNet`.
- Input: [B,512,8] PyTorch; [1,8,512] exported TFLite.
- Preprocessing: Replay crops + channel-zscores; exported path transposes [T,C] to [C,T].
- Topology: temporal Conv2D -> BN -> depthwise spatial Conv2D -> BN -> ELU -> AvgPool -> Dropout -> separable temporal Conv2D -> pointwise Conv2D -> BN -> ELU -> AvgPool -> Dropout -> adaptive avg pool -> Linear
- Operators: AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, ELU, FULLY_CONNECTED, MEAN, PAD, RESHAPE, STRIDED_SLICE.
- Parameters: 1436; artifact size: 11.5 KiB.
- Accumulation-heavy components: long temporal convolution, depthwise convolutions, average pooling, final dense layer
- Risk classification: moderate (Small logit drift can still flip low-margin decisions; topology alone does not identify the layer.)

## EEGNet-int8

- Model: `src/models/eegnet.py` / `EEGNet`.
- Input: [B,512,8] PyTorch; [1,8,512] exported TFLite.
- Preprocessing: Replay crops + channel-zscores; exported path transposes [T,C] to [C,T].
- Topology: temporal Conv2D -> BN -> depthwise spatial Conv2D -> BN -> ELU -> AvgPool -> Dropout -> separable temporal Conv2D -> pointwise Conv2D -> BN -> ELU -> AvgPool -> Dropout -> adaptive avg pool -> Linear
- Operators: AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, DEQUANTIZE, ELU, FULLY_CONNECTED, MEAN, PAD, QUANTIZE, RESHAPE, STRIDED_SLICE.
- Parameters: 1436; artifact size: 10.7 KiB.
- Accumulation-heavy components: long temporal convolution, depthwise convolutions, average pooling, final dense layer
- Risk classification: moderate (Small logit drift can still flip low-margin decisions; topology alone does not identify the layer.)

## FBtCNN-f32

- Model: `src/models/fbtcnn.py` / `FBtCNN`.
- Input: [B,512,8] PyTorch; [1,8,512,4] exported TFLite after external 4-band filterbank.
- Preprocessing: Replay crops + channel-zscores; external four-band Butterworth filtfilt before TFLite.
- Topology: 4 filter-bank branches -> shared spatial Conv2D/BN/ELU/Dropout -> temporal Conv2D/BN/ELU/Dropout -> temporal Conv2D/BN/ELU/Dropout -> sum fusion -> final Conv2D/BN/ELU/Dropout -> Flatten -> Linear
- Operators: CONV_2D, ELU, FULLY_CONNECTED, PAD, RESHAPE, SUM, TRANSPOSE.
- Parameters: 183292; artifact size: 720.5 KiB.
- Accumulation-heavy components: four IIR filtfilt branches, branch CNNs, branch-sum fusion, final temporal convolution
- Risk classification: high (Multi-branch filtering and long temporal convolutions are consistent with high latency; observed int8 flips remain rare.)

## FBtCNN-int8

- Model: `src/models/fbtcnn.py` / `FBtCNN`.
- Input: [B,512,8] PyTorch; [1,8,512,4] exported TFLite after external 4-band filterbank.
- Preprocessing: Replay crops + channel-zscores; external four-band Butterworth filtfilt before TFLite.
- Topology: 4 filter-bank branches -> shared spatial Conv2D/BN/ELU/Dropout -> temporal Conv2D/BN/ELU/Dropout -> temporal Conv2D/BN/ELU/Dropout -> sum fusion -> final Conv2D/BN/ELU/Dropout -> Flatten -> Linear
- Operators: CONV_2D, DEQUANTIZE, ELU, FULLY_CONNECTED, PAD, QUANTIZE, RESHAPE, SUM, TRANSPOSE.
- Parameters: 183292; artifact size: 188.3 KiB.
- Accumulation-heavy components: four IIR filtfilt branches, branch CNNs, branch-sum fusion, final temporal convolution
- Risk classification: high (Multi-branch filtering and long temporal convolutions are consistent with high latency; observed int8 flips remain rare.)
