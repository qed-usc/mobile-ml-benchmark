# Inference Latency Measurements for CNNs

## File Descriptions

The files in this folder follow the naming convention `{framework}_cnn_{dataset}_{info}.csv`, where each part of the filename represents the following:

- **framework**: Specifies the framework used for inference.
  - `tflite`: TensorFlow Lite framework.
  - `torch`: PyTorch Mobile framework.

- **dataset**: Specifies the type of neural architectures.
  - `real`: Real-world CNNs.
  - `synthetic`: Synthetic CNNs sampled from a NAS search space.

- **info**: Specifies the type of latency measured.
  - `e2e`: End-to-end latency, representing the total inference time for processing the input through the entire model.
  - `ops_cpu`: Operation-wise latency on mobile CPUs using 32-bit floating-point precision, detailing the time taken by each operation when running the model on a mobile CPUs.
  - `ops_cpu_quant`: Operation-wise latency on mobile CPUs using 8-bit integer quantization, detailing the time taken by each operation when running the model on a mobile CPUs.
  - `ops_gpu`: Operation-wise latency on mobile GPUs using 16-bit floating-point precision, detailing the time taken by each operation when running the model on a mobile GPUs.

### Example Filenames:
- `tflite_cnn_real_e2e.csv`: End-to-end latency measurements for real-world neural architectures using TensorFlow Lite.
- `tflite_cnn_synthetic_ops_cpu.csv`: Operation-wise latency on mobile CPUs for synthetic neural architectures using TensorFlow Lite.
