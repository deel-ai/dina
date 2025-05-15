# 🎯 DINA: DEEL ImageNet Attributions

Welcome to the official repository for **DINA (DEEL ImageNet Attributions)** — a dataset and benchmark framework for evaluating explanation methods on models that have been trained on ImageNet.

This repository provides:

- 📦 Code to generate attribution maps on ImageNet
- 🧪 Scripts to compute explanation **metrics** (Fidelity, Complexity, Randomization)
- 🧠 A curated dataset of **precomputed attributions** hosted in a public GCS bucket

> 📂 All attribution files are available at:  
> **`https://storage.cloud.google.com/xai-deel`**

<details>
<summary>🧭 Attribution Coverage Table (Click to expand)</summary>

<br>

This dataset includes attribution maps for all models and explainers listed below:

| Model         | Saliency | GradCAM | GradCAMPP | VarGrad | SmoothGrad | SquareGrad | IntegratedGradients | Rise | GradientInput | KernelShap | Occlusion | Lime | HsicAttributionMethod | SobolAttributionMethod |
|--------------|----------|---------|-----------|---------|-------------|-------------|----------------------|------|----------------|-------------|-----------|------|------------------------|------------------------|
| BeitV2       | ✅       | ❌      | ❌        | ✅      | ✅          | ✅          | ✅                   | ✅   | ✅             | ✅          | ✅        | ✅   | ✅                     | ✅                     |
| ConvNeXtV2   | ✅       | ✅      | ✅        | ✅      | ✅          | ✅          | ✅                   | ✅   | ✅             | ✅          | ✅        | ✅   | ✅                     | ✅                     |
| DinoV2       | ✅       | ❌      | ❌        | ✅      | ✅          | ✅          | ✅                   | ✅   | ✅             | ✅          | ✅        | ✅   | ✅                     | ✅                     |
| EfficientNetV2 | ✅     | ✅      | ✅        | ✅      | ✅          | ✅          | ✅                   | ✅   | ✅             | ✅          | ✅        | ✅   | ✅                     | ✅                     |
| InceptionNeXt| ✅       | ✅      | ✅        | ✅      | ✅          | ✅          | ✅                   | ✅   | ✅             | ✅          | ✅        | ✅   | ✅                     | ✅                     |
| MaxVIT       | ✅       | ❌      | ❌        | ✅      | ✅          | ✅          | ✅                   | ✅   | ✅             | ✅          | ❌        | ✅   | ✅                     | ✅                     |
| MLPMixer     | ✅       | ❌      | ❌        | ✅      | ✅          | ✅          | ✅                   | ✅   | ✅             | ✅          | ✅        | ✅   | ✅                     | ✅                     |
| ResNest50    | ✅       | ✅      | ✅        | ✅      | ✅          | ✅          | ✅                   | ✅   | ✅             | ✅          | ✅        | ✅   | ✅                     | ✅                     |
| ResNet50     | ✅       | ✅      | ✅        | ✅      | ✅          | ✅          | ✅                   | ✅   | ✅             | ✅          | ✅        | ✅   | ✅                     | ✅                     |

</details>

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone git@github.com:deel-ai/dina.git
cd dina
```

### 2️⃣ Set up a Python Environment

```bash
python3.11 -m venv dina_env
source dina_env/bin/activate
pip install -e .
```

### 📥 Prepare the ImageNet Dataset

1. You need to download the [ImageNet dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).
2. The validation labels must be downloaded from the tensorflow research models [repo](https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt).
3. The validation labels must be placed in the ILSVRC/Data/CLS-LOC folder.
4. Structure the Validation Set

```bash
PYTHONPATH=. python scripts/structure_val_imagenet.py --raw_dir "path/to/imagenet/ILSVRC/Data/CLS-LOC"
```

### 🧩 Dataset Metadata

**Generate `imagenet_data.json`**

We use Keras CV Attention Models [(Kecam)](https://github.com/leondgarse/keras_cv_attention_models/tree/main) to load vision models. You’ll need to create a metadata file:

```bash
PYTHONPATH=. python scripts/kacem_custom_dataset_script.py \
  --train_images path/to/imagenet/ILSVRC/Data/CLS-LOC/train \
  --test_images /path/to/imagenet/ILSVRC/Data/CLS-LOC/val \
  -s imagenet_data
```

Place the resulting `imagenet_data.json` in the root of the repo.

## 🖼️ Generate Explanations

Example to run IntegratedGradients on DinoV2:

```bash
PYTHONPATH=. python scripts/compute_explanations.py \
  --model_name "DinoV2" \
  --explainer_name "IntegratedGradients" \
  --batch_size 32 \
  --imagenet_json imagenet_data.json \
  --output_dir /dir/to/bench_dir/
```
📍 See: deel/dina/utils/models.py

📍 See: deel/dina/utils/explainers.py

Attributions are saved at:

```bash
{output_dir}/{model_name}/{explainer_name}/explanations.tfrecord
```

A preprocessed dataset (~30GB) will also be generated and saved to:

```bash
{output_dir}/{model_name}/preprocess_dataset.tfrecord
```

> [!NOTE]
> If you run explanations with the same model but another explainer it will directly reload the preprocess_dataset!

## 📊 Benchmark Metrics

### 🔹 Fidelity Metrics (Insertion, Deletion, MuFidelity)
```bash
PYTHONPATH=. python scripts/compute_fidelity.py \
  --model_name "DinoV2" \
  --explainer_name "IntegratedGradients" \
  --metrics "Insertion" "Deletion" \
  --batch_size 32 \
  --imagenet_json imagenet_data.json \
  --output_dir /dir/to/bench_dir/
```
📍 See: deel/dina/metrics/fidelity.py

### 🔹 Complexity Metrics (Sparseness, Complexity)
```bash
PYTHONPATH=. python scripts/compute_complexity.py \
  --model_name "DinoV2" \
  --explainer_name "IntegratedGradients" \
  --metrics "Sparseness" "Complexity" \
  --batch_size 32 \
  --imagenet_json imagenet_data.json \
  --output_dir /dir/to/bench_dir/
```
📍 See: deel/dina/metrics/complexity.py

### 🔹 Randomization Metrics
```bash
PYTHONPATH=. python scripts/compute_randomization.py \
  --model_name "DinoV2" \
  --explainer_name "IntegratedGradients" \
  --metric_name "ModelRandomizationMetric05" \
  --batch_size 32 \
  --imagenet_json imagenet_data.json \
  --output_dir /dir/to/bench_dir/
```
📍 See: deel/dina/metrics/randomization.py

`metric_name` options: "ModelRandomizationMetric05", "ModelRandomizationMetric01", "RandomLogitMetric"

> [!NOTE]
> Due to the fact we randomize the model, we cannot run several metrics at once for those ones.

## 🧾 Load Attributions with TensorFlow

For attributions that you have downloaded from our bucket (or that you have computed) you can load them as follow:

```python
from deel.dina.utils import create_explanations_dataset_from_tfrecord

explanations_path = "/path/to/explanations.tfrecord"
explanations_ds = create_explanations_dataset_from_tfrecord(explanations_path)
```

To get the input-label-explanation triplet:

> [!IMPORTANT]
> This will create a tfrecord file at output_dir which will be aroung 30Go

```python
from deel.dina.utils import get_model, generate_or_load_preprocess_ds
import tensorflow as tf

model = get_model("BeitV2")
preprocess_dataset = generate_or_load_preprocess_ds(
    model=model,
    output_dir="path/to/preprocess_record_dir",
    batch_size=32,
    imagenet_json_path="imagenet_data.json"
)

combined_ds = tf.data.Dataset.zip((preprocess_dataset, explanations_ds))
combined_ds = combined_ds.map(lambda x, y: (x[0], x[1], y))

for pre_input, one_hot_pred, explanation in combined_ds.batch(8):
    # Your logic here
    break
```