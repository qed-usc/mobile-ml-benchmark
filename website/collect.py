import pandas as pd


def gen_framework():
    framework_data = [
        ['TensorFlow Lite', 'v2.10.0', "https://github.com/165749/tensorflow"],  # For CNN
        ['TensorFlow Lite', 'v2.15.0', "https://github.com/tensorflow/tensorflow/tree/v2.15.0"],  # For ViT
        ['PyTorch Mobile', 'v2.0.0', "https://github.com/165749/pytorch"],
    ]
    df = pd.DataFrame(framework_data, columns=['frameworkName', 'frameworkVersion', 'implementationURL'])
    df.to_csv('framework.csv', sep='\t', index=False)


def gen_device():
    device_data = [
        ['Google Pixel 4', 'Snapdragon 855'],
        ['Xiaomi Mi 8 SE', 'Snapdragon 710'],
        ['Samsung Galaxy S10', 'Exynos 9820'],
        ['Samsung Galaxy A03s', 'Helio P35'],
        ['Apple iPhone XS', 'A12 Bionic'],
        ['Apple iPhone 7', 'A10 Fusion'],
    ]
    df = pd.DataFrame(device_data, columns=['deviceName', 'socName'])  # TODO (Zhuojin): RAM
    df.to_csv('device.csv', sep='\t', index=False)


def gen_SoC():
    soc_data = [
        ['Snapdragon 855', 1, 2.84, 3,    2.32, 4, 1.80, 'Adreno 640'],
        ['Snapdragon 710', 2, 2.20, None, None, 6, 1.70, 'Adreno 616'],
        ['Exynos 9820',    2, 2.73, 2,    2.31, 4, 1.95, 'Mali G76'],
        ['Helio P35',      4, 2.30, None, None, 4, 1.80, 'PowerVR GE8320'],
        ['A12 Bionic',     2, 2.49, None, None, 4, 1.52, 'Apple-designed G11P'],
        ['A10 Fusion',     2, 2.34, None, None, 2, 1.05, 'PowerVR GT7600 Plus (Custom)'],
    ]
    df = pd.DataFrame(
        soc_data,
        columns=['socName', 'largeNum', 'largeFreq(GHz)', 'medNum', 'medFreq(GHz)', 'smallNum', 'smallFreq(GHz)', 'gpuModel']  # TODO (Zhuojin): gpuFreq(MHz)
    )
    df.to_csv('soc.csv', sep='\t', index=False)


def gen_model():
    dataset_list = [
        ('../dataset/torch_cnn_e2e.csv', 'CNN', False),
        ('../dataset/torch_nas_e2e.csv', 'CNN', True),
        ('../dataset/torch_transformer_e2e.csv', 'ViT', False),
    ]
    df_list = []
    for path, category, is_synthetic in dataset_list:
        df = pd.read_csv(path)
        df = df[['model', 'flops', 'size']]
        df['type'] = category
        df['synthetic'] = is_synthetic
        df_list.append(df)
    df = pd.concat(df_list)
    df.to_csv('model.csv', sep='\t', index=False)


def gen_config():
    frameworks = [
        'TensorFlow Lite v2.10',
        'TensorFlow Lite v2.15',
        'PyTorch Mobile v2.0.0',
    ]
    hardware_config = {
        'Snapdragon 855': ['1large', '1med', '2med', '3med', '1large1med', '1small', '2small', '3small', '4small'],
        'Snapdragon 710': ["1large", "2large", "1small", "2small", "3small", "4small"],
        'Exynos 9820': ["1large", "2large", "1med", "2med", "1large2med", "1small", "2small", "3small", "4small"],
        'Helio P35': ["1large", "2large", "3large", "4large", "1small", "2small", "3small", "4small"],
        'A12 Bionic': ["1large", "2large", "1small", "2small", "3small", "4small"],
        'A10 Fusion': ["1large", "2large"],
    }
    precisions = ['float32', 'int8']

    config_mappings = {}
    config_data = []
    for soc, core_list in hardware_config.items():
        for core in core_list:
            for precision in precisions:
                for framework in frameworks:
                    config_id = len(config_data)
                    config_data.append([config_id, soc, f'CPU({core})', precision, framework])
                    config_mappings[(soc, f'CPU({core})', precision, framework)] = config_id
        for framework in frameworks:
            config_id = len(config_data)
            config_data.append([config_id, soc, 'GPU', 'float32', framework])
            config_mappings[(soc, f'GPU', 'float32', framework)] = config_id

    df = pd.DataFrame(config_data, columns=['configID', 'socName', 'hardware', 'precision', 'framework'])
    df.to_csv('config.csv', sep='\t', index=False)
    return config_mappings


def gen_e2e(config_mappings):
    device_to_soc = {
        'pixel4': 'Snapdragon 855',
        'onefusion': 'Snapdragon 710',
        'mi8se': 'Snapdragon 710',
        's10': 'Exynos 9820',
        'a03s': 'Helio P35',
        'iphonexs': 'A12 Bionic',
        'iphone7': 'A10 Fusion',
    }
    dataset_list = [
        ('../dataset/torch_cnn_e2e.csv', 'PyTorch Mobile v2.0.0'),
        ('../dataset/torch_nas_e2e.csv', 'PyTorch Mobile v2.0.0'),
        ('../dataset/torch_transformer_e2e.csv', 'PyTorch Mobile v2.0.0'),
        ('../dataset/tflite_cnn_e2e.csv', 'TensorFlow Lite v2.10'),
        ('../dataset/tflite_nas_e2e.csv', 'TensorFlow Lite v2.10'),
        ('../dataset/tflite_transformer_e2e.csv', 'TensorFlow Lite v2.15'),
    ]
    df_list = []
    for path, framework in dataset_list:
        df = pd.read_csv(path)
        core_configs = [x for x in df.columns.to_list() if x.endswith('|mean')]
        for core_config in core_configs:
            device, hardware, precision, _ = core_config.split('|')
            hardware = 'GPU' if hardware == 'gpu' else f'CPU({hardware})'
            precision = 'float32' if precision == 'float' else 'int8'
            config = (device_to_soc[device], hardware, precision, framework)
            if config not in config_mappings:
                print(f'Warning: [e2e_latency.csv] Skip configuration {config}!')
                continue
            config_id = config_mappings[config]
            new_df = df[['model', core_config]].copy()
            new_df = new_df.rename(columns={core_config: 'latency'})
            new_df.insert(1, 'configID', config_id)
            new_df = new_df.dropna()
            df_list.append(new_df)
    df = pd.concat(df_list)
    df.to_csv('e2e_latency.csv', sep='\t', index=False)


def gen_ops(config_mappings):
    device_to_soc = {
        'pixel4': 'Snapdragon 855',
        'onefusion': 'Snapdragon 710',
        'mi8se': 'Snapdragon 710',
        's10': 'Exynos 9820',
        'a03s': 'Helio P35',
        'iphonexs': 'A12 Bionic',
        'iphone7': 'A10 Fusion',
    }
    dataset_list = [
        ('../dataset/torch_cnn_ops_cpu.csv', 'PyTorch Mobile v2.0.0'),
        ('../dataset/torch_cnn_ops_cpu_quant.csv', 'PyTorch Mobile v2.0.0'),
        ('../dataset/torch_cnn_ops_gpu.csv', 'PyTorch Mobile v2.0.0'),
        ('../dataset/torch_nas_ops_cpu.csv', 'PyTorch Mobile v2.0.0'),
        ('../dataset/torch_nas_ops_cpu_quant.csv', 'PyTorch Mobile v2.0.0'),
        ('../dataset/torch_nas_ops_gpu.csv', 'PyTorch Mobile v2.0.0'),
        ('../dataset/torch_transformer_ops_cpu.csv', 'PyTorch Mobile v2.0.0'),
        ('../dataset/torch_transformer_ops_cpu_quant.csv', 'PyTorch Mobile v2.0.0'),
        ('../dataset/tflite_cnn_ops_cpu.csv', 'TensorFlow Lite v2.10'),
        ('../dataset/tflite_cnn_ops_gpu.csv', 'TensorFlow Lite v2.10'),
        ('../dataset/tflite_nas_ops_cpu.csv', 'TensorFlow Lite v2.10'),
        ('../dataset/tflite_nas_ops_gpu.csv', 'TensorFlow Lite v2.10'),
        ('../dataset/tflite_transformer_ops_cpu.csv', 'TensorFlow Lite v2.15'),
    ]
    feature_df_list = []
    df_list = []
    feature_id_offset = 0
    for path, framework in dataset_list:
        df = pd.read_csv(path)
        core_configs = [x for x in df.columns.to_list() if x.endswith('float') or x.endswith('quant')]
        df.insert(0, 'featureID', range(feature_id_offset, len(df) + feature_id_offset))
        df.insert(0, 'Empty', 'N/A')
        feature_id_offset += len(df)
        feature_df_list.append(df[['featureID', 'model', 'operation', 'feature']].copy())
        for core_config in core_configs:
            device, hardware, precision = core_config.split('|')
            hardware = 'GPU' if hardware == 'gpu' else f'CPU({hardware})'
            precision = 'float32' if precision == 'float' else 'int8'
            config = (device_to_soc[device], hardware, precision, framework)
            if config not in config_mappings:
                print(f'Warning: [ops_latency.csv] Skip configuration {config}!')
                continue
            config_id = config_mappings[config]
            if 'GPU' in hardware:
                kernel_column = f'{device}|gpu|kernel'
            else:
                kernel_column = 'Empty'
            new_df = df[['model', 'operation', 'featureID', kernel_column, core_config]].copy()
            new_df = new_df.rename(columns={core_config: 'latency', kernel_column: 'gpuKernel'})
            new_df.insert(2, 'configID', config_id)
            new_df = new_df.dropna()
            df_list.append(new_df)
    df = pd.concat(df_list)
    feature_df = pd.concat(feature_df_list)
    df.to_csv('ops_latency.csv', sep='\t', index=False)
    feature_df.to_csv('ops_feature.csv', sep='\t', index=False)


def main():
    gen_device()
    gen_SoC()
    gen_framework()
    gen_model()
    config_mappings = gen_config()
    gen_e2e(config_mappings)
    gen_ops(config_mappings)


if __name__ == '__main__':
    main()
