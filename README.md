## 1. Sky Image Descriptor

The sky image descriptor (SID) leverages the Sky Finder dataset [1], which contains a rich variety of sky imagery. We manually categorized the 20 most relevant scenes into three distinct classes: clear, partial, and overcast, based on sky conditions. Using this classified data, we trained a ResNet50 backbone [2] with a multi-layer perceptron head. The model was trained on a contrastive learning task, enabling it to extract meaningful sky image representations from the diverse sky conditions present in the dataset.



### 1.1 Datasets

#### 1.1.1 Sky Finder Dataset

The Sky Finder dataset comprises high-resolution outdoor images captured across various locations, weather conditions, and times of day. Our preprocessing involves:

1. **Image Classification**: We manually categorized the 20 most representative scenes in the dataset into three classes based on sky visibility, yielding 21,490 images across the three classes:
    - **Clear**: (6,335 images) Scenes with predominantly visible blue sky and minimal cloud coverage.
    - **Partial**: (6,378 images) Scenes with mixed cloud and clear sky regions.
    - **Overcast**: (8,777 images) Scenes with complete or near-complete cloud coverage.
2. **Image Preprocessing**: Images are cropped based on manually labeled ground segmentation to remove non-sky regions, and then in-painted using TELEA algorithm [3] with a radius of 3 pixels to seamlessly fill any artifacts along the segmentation boundary.

For experimental evaluation, the dataset is divided into training, validation, and test sets containing 12,894 (60%), 4,298 (20%), and 4,298 (20%) images, respectively.

#### 1.1.2 Sky Finder Cover Dataset

The Sky Finder Cover Dataset is a manually annotated subset of the Sky Finder Dataset with pixel-level cloud segmentation masks. This carefully curated dataset maintains the same classification schema (clear, partial, and overcast) as the original Sky Finder Dataset, providing high-quality ground truth for cloud segmentation tasks.

The dataset was created through a meticulous annotation process:
1. **Selection**: Representative images were selected from each sky condition category to ensure diversity.
2. **Manual Segmentation**: Annotators created pixel-precise binary masks, where each pixel is labeled as either overcast (white), partially covered (gray) or clear sky/ground (0).

For experimental evaluation, the dataset was divided into training and validation sets containing 182 and 58 images, respectively.

#### 1.1.2 Pair Generation for Contrastive Learning

Our contrastive learning framework relies on creating meaningful sample pairs:

1. **Positive Pairs**: For each processed image in the dataset, we generate two different augmented views of the same base image. These views are created through a series of transformations aiming to keep the core content of the image intact while introducing variability.

2. **Negative Pairs**: All other augmented views from different base images in the batch serve as negative examples. The model learns to distinguish these from the positive pairs.

<div align="center">
    <img src="generated/pair_generation.png" alt="Pair generation process" align="center" width="80%">
    <div align="center">
    <em>Figure 1: Pair generation process for contrastive learning. Each original image is cropped to remove the ground region, inpainted and augmented to create two images, which are then used as positive pairs.</em>
    </div>
</div>



### 1.2 Model Architecture

#### 1.2.1 SID Backbone Network

The SID model employs a ResNet50 backbone pretrained on ImageNet [4] as the feature encoder, with the original classification head replaced by a projection head. The projection head consists of a two-layer multi-layer perceptron (MLP) with ReLU activation between layers, mapping the 2048-dimensional ResNet50 feature vector to a 16-dimensional SID space. The final descriptors are L2-normalized.

#### 1.2.2 Classification Head for Downstream Validation

To evaluate the quality of learned SID representations, we implement a simple classification head consisting of a 3-layer fully connected network. The architecture includes:

- **Input Layer**: Accepts 16-dimensional SID embeddings
- **Hidden Layers**: Two fully connected layers with ReLU activations.
- **Output Layer**: 3-way linear layer producing probabilities for clear, partial, and overcast sky conditions.

This lightweight classification head serves as a downstream task to validate that the learned SID representations capture semantically meaningful sky condition features.


### 1.3 Training Objective

#### 1.3.1 Contrastive Learning Objective

We employ the Normalized Temperature-scaled Cross Entropy (NT-Xent) loss, which is formulated as:

$$\mathcal{L} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N}\mathbf{1}_{[k \neq i]}\exp(\text{sim}(z_i, z_k)/\tau)}$$

Where:
- $z_i$ and $z_j$ are normalized descriptors of two augmented views of the same image.
- $\text{sim}(u, v)$ denotes the cosine similarity between vectors $u$ and $v$.
- $\tau$ is a temperature parameter that controls the concentration level of the distribution.
- $N$ is the number of image pairs in the current batch.
- $\mathbf{1}_{[k \neq i]}$ is an indicator function that equals 1 when $k \neq i$.

This loss function encourages the model to learn representations where similar samples are pulled together in the descriptor space while dissimilar samples are pushed apart, resulting in a model that effectively captures the distinctive characteristics of different sky conditions.

#### 1.3.2 Classification Head Training Objective

The classification head is trained using standard cross-entropy loss:

$$L_{cls} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

Where $C=3$ represents the number of sky condition classes, $y_i$ is the ground truth label, and $\hat{y}_i$ is the predicted probability for class $i$. The classification head is trained separately after the SID backbone has been trained and frozen.



### 1.4 Training Procedure

#### 1.4.1 SID Backbone Training

Our SID model was trained with the following hyperparameters and configuration:

- **Optimizer**: AdamW with a learning rate of $10^{-4}$ and weight decay of $10^{-4}$.
- **Embedding Dimension**: 16 (latent space dimension at the end of the MLP head).
- **Batch Configuration**: 2 batches with 3 pairs per batch ($N=3$).
- **Training Duration**: 4 epochs.
- **Temperature Parameter**: 0.5 for the NT-Xent loss.
- **Learning Rate Scheduler**: Reduce learning rate on plateau with a patience of 1 epoch and a factor of 0.5.
- **Hardware**: Single NVIDIA RTX 3080 GPU with 10GB of memory.

#### 1.4.2 Classification Head Training

The classification head training follows a standard supervised learning approach:

- **Optimizer**: AdamW with a learning rate of $10^{-3}$ and weight decay of $10^{-4}$.
- **Batch Size**: 32 images.
- **Training Duration**: 100 epochs with early stopping based on validation loss.
- **Learning Rate Scheduler**: Reduce learning rate on plateau with a patience of 1 epoch and a factor of 0.5.

These configurations provide a good balance between performance and computational efficiency, allowing the models to learn meaningful representations while remaining trainable on consumer-grade hardware.



### 1.5 Results

#### 1.5.1 Sky Image Descriptor Space Visualization

The trained SID model is evaluated on the Sky Finder dataset, and the results are visualized using UMAP [5]. The resulting plots demonstrate how the model effectively clusters similar sky conditions together in the descriptor space.

Figure 2a shows the sky image descriptor space visualization grouped by semantic sky class labels (clear, partial, overcast), revealing natural clustering of similar sky conditions. Remarkably, when applying unsupervised K-means clustering to the same descriptor space (Figures 2b-c), the resulting clusters closely follow the boundaries of the semantic sky classes. This alignment between unsupervised clustering and human-interpretable labels demonstrates that the SID model has learned meaningful representations that capture real physical and visual patterns in sky conditions without requiring explicit supervision during the descriptor extraction phase.

The K-means clustering reveals distinct patterns that correspond to recognizable sky characteristics. The rightmost cluster (blue in K=3, orange in K=4) centers on clear sky conditions, capturing images with predominantly blue skies and minimal cloud coverage. The bottom-center cluster (green in K=3, blue in K=4) focuses mostly on partially cloudy conditions and highly textured overcast skies, encompassing a diverse range of sky patterns from delicate veil clouds and scattered cumulus formations to heavily cloudy skies with significant texture variation.

In the K=3 clustering (Figure 2b), the remaining cluster primarily contains overcast skies. However, the K=4 clustering (Figure 2c) reveals a more nuanced structure by splitting overcast conditions into two distinct subclusters. The green cluster (leftmost region) centers around heavily and uniformly overcast skies with minimal texture variation, representing completely cloud-covered conditions or fog. In contrast, the red cluster (topmost region) captures overcast skies with more visual texture and contrast, likely including scenes where sky partially penetrates through cloud layers or where cloud formations exhibit greater structural variation. This finer granularity suggests that the SID representations encode subtle but meaningful differences in cloud density, texture, and spatial patterns that align with human visual perception of sky conditions.

<div align="center">
  <img src="generated/sky_image_descriptor_space/sky_image_descriptor_space_sky_class.png" alt="Sky Image Descriptor Space - Sky Class Grouping" width="90%">
  <br>
  <em><strong>Figure 2a:</strong> Sky image descriptor space visualization grouped by semantic sky class labels (clear, partial, overcast).</em>
</div>

<br>

<div align="center">
  <table>
      <tr>
          <td align="center">
              <img src="generated/sky_image_descriptor_space/sky_image_descriptor_space_cluster_3.png" alt="Sky Image Descriptor Space - 3 Clusters" width="100%">
              <br>
              <em><strong>Figure 2b:</strong> K-means clustering with K=3</em>
          </td>
          <td align="center">
              <img src="generated/sky_image_descriptor_space/sky_image_descriptor_space_cluster_4.png" alt="Sky Image Descriptor Space - 4 Clusters" width="100%">
              <br>
              <em><strong>Figure 2c:</strong> K-means clustering with K=4</em>
          </td>
      </tr>
  </table>
  <em><strong>Figure 2:</strong> UMAP visualization of the trained SID model on the Sky Finder dataset showing both semantic groupings and unsupervised clustering patterns in the sky image descriptor space.</em>
</div>

#### 1.5.2 Downstream Classification Performance

To quantitatively validate the quality of the learned SID representations, we evaluate their performance on the downstream task of sky condition classification. The 16-dimensional SID embeddings are fed into the classification head described in Section 1.2.2, and the model is trained to predict the three sky condition classes.

The classification results demonstrate good performance across all evaluation metrics. The SID representations achieve over 86% accuracy on the test set with minimal performance degradation between training and test splits, indicating strong generalization capabilities. The high F1 scores across all splits confirm that the learned representations capture discriminative features that enable accurate sky condition classification, providing quantitative validation of the semantic clustering patterns observed in the UMAP visualizations.

<div align="center">
   <em><strong>Table 1:</strong> Classification performance of the SID-based sky condition classifier. Results demonstrate high accuracy and F1 scores across all data splits, validating the quality of learned representations.</em>
</div>

<div align="center">

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 0.8609 | 0.8616 | 0.8637 |
| **F1 Score** | 0.8584 | 0.8561 | 0.8590 |

</div>

Analysis of the confusion matrix reveals interesting patterns in per-class performance. The clear sky class achieves the highest F1 score (0.9040), followed by overcast conditions (0.8904), while partial sky conditions show the lowest performance (0.7826). This performance disparity likely stems from the inherently ambiguous nature of partial sky conditions, which represent a transitional state between clear and overcast skies. The non-binary characteristics of partial conditions create classification challenges at decision boundaries, where distinguishing between clear-partial or overcast-partial transitions becomes ambiguous. This finding aligns with the expected difficulty in categorizing intermediate sky states and highlights the model's stronger performance on more distinctive sky conditions with clearer visual characteristics.

### 1.6 Plotting New Data in the SID Space

The SID space was built on the sky finder dataset, but new sky images can be projected into the SID space using the trained SID model. This allows for the visualization of new sky images in the same descriptor space, enabling comparison between new images or between sky finder manually labelled images.

#### 1.6.1 Window View Dataset

Adapting our own dataset of window views introduced in [6], we can pass the images through the SID model to obtain their sky image descriptors. The window view dataset contains 45 high-resolution images captured from fifteen different locations across the EPFL campus in Switzerland between March and May 2023, encompassing a wide range of atmospheric conditions including clear, partial, and overcast skies.

The images were captured using a calibrated Canon EOS R5 DSLR camera with dual fisheye lens at 6K resolution (6144×3072 pixels), then converted to 180-degree equirectangular projection format. To ensure physically accurate window view representation, each scene was captured alongside a 1:10 scale model of an office room with horizontal aperture. The capture locations maintained a minimum 6-meter distance from moving objects and aimed for balanced visual composition with approximately 25% greenery and 40% sky-to-window ratio, following European Daylight Standard EN17037 criteria.

This dataset was originally developed for virtual reality research investigating how dynamic versus static window view representations affect visual perception and building occupant experience. When processed through our trained SID model, these real-world window view images provide valuable validation of the sky image descriptor space learned from the Sky Finder dataset, enabling evaluation of how the model generalizes to practical architectural viewing scenarios.

#### 1.6.2 Methodology

To project new sky images into the trained SID space, we developed a processing pipeline that ensures consistent representation with the Sky Finder dataset. The methodology follows these sequential steps:

1. **Manual Image Cropping:** For images containing specific viewing contexts (e.g., window views), manual cropping can be applied to focus on the desired region. This step utilizes manually annotated binary masks to define the region of interest, ensuring that only relevant content is analyzed while excluding irrelevant elements.

2. **Sky Region Detection and Cropping:** The sky region is automatically segmented using Grounded Segment Anything 2 (GSAM2) [7] with the keyword prompt "sky". This state-of-the-art segmentation model provides accurate and robust segmentation of sky regions in images, even under challenging conditions such as varying lighting or atmospheric effects, and it eliminates the need for manual annotation of new datasets. Following segmentation, the image is automatically cropped to the bounding box of the detected sky region, removing non-sky areas and focusing the analysis on the relevant atmospheric content.

3. **Boundary Artifact Removal:** The cropped sky region undergoes inpainting using the TELEA algorithm with a radius of 3 pixels. This step removes segmentation boundary artifacts and ensures smooth transitions at mask boundaries, preserving the integrity of the sky region while eliminating potential artifacts that could affect descriptor quality.

#### 1.6.3 Results

The projection of window view images into the trained SID space demonstrates successful generalization of the learned sky descriptors to real-world architectural viewing scenarios. Figure 3 shows the UMAP visualization where the 42 window view images (represented as crosses) are distributed throughout the descriptor space alongside the Sky Finder dataset points.

The window view images exhibit meaningful spatial distribution within the established sky condition clusters. Clear sky conditions from the window views (blue crosses) predominantly map to the rightmost region of the descriptor space, aligning with the clear sky cluster learned from the Sky Finder dataset. Partial sky conditions (orange crosses) distribute primarily in the central regions, overlapping with the mixed cloud and clear sky patterns identified during training. Overcast conditions (red crosses) cluster in the left portion of the space, corresponding to the heavily clouded regions established by the original training data.

<div align="center">
  <img src="generated/sky_image_descriptor_space/sky_image_descriptor_space_oos.png" alt="Sky Image Descriptor Space - Sky Class Grouping" width="90%">
  <br>
  <em><strong>Figure 3:</strong> Sky image descriptor space visualization with new window view images projected into the SID space. The new images are represented as crosses, with colors indicating their estimated sky condition class (blue for clear, orange for partial and red for overcast).</em>
</div>

### 1.7 Reproduction Procedure

Follow these steps to reproduce our SID results by generating the dataset, training the model and plotting the SID space.

#### 1.7.1 Sky Finder Dataset Generation

To prepare the dataset for training, execute the following commands which will download and organize the Sky Finder images according to our classification schema:

```bash
cd src/datasets
python generate_sky_finder_dataset.py [-w <max-workers>] [-f] [-r]
```

**Parameters:**
- `-w`, `--max-workers`: (Optional, default: 3) Specifies the maximum number of concurrent workers for downloading images. Higher values speed up the download process but require more system resources.
- `-f`, `--force`: (Optional, default: false) Forces re-download, re-extraction, re-classification, and re-splitting of data even if it already exists locally, ensuring you have the latest version.
- `-r`, `--remove-data`: (Optional, default: false) Automatically removes temporary archives and extracted data after processing (keeps final split data) to save disk space.

#### 1.7.2 Training the SID Model

To train the SID model, execute the following commands:

```bash
cd src/contrastive_net
python contrastive_net_train.py [-e <epochs>] [-b <batch-size>] [-w <workers>] [-evaluation-steps <evaluation-steps>] [--learning-rate <learning-rate>] [--weight-decay <weight-decay>] [--project-name <project-name>] [--experiment-name <experiment-name>] [--accelerator <accelerator>] [--devices <devices>] [--precision <precision>] [--save-top-k <save-top-k>] [--no-pretrained] [--no-normalize]
```

**Parameters:**
- `-e`, `--epochs`: (Optional, default: 4) Number of training epochs.
- `-b`, `--batch-size`: (Optional, default: 2) Batch size for training.
- `-w`, `--n-workers`: (Optional, default: 8) Number of data loading workers.
- `--evaluation-steps`: (Optional, default: 500) Number of steps between validation runs.
- `--learning-rate`: (Optional, default: 1e-4) Learning rate for optimization.
- `--weight-decay`: (Optional, default: 1e-4) Weight decay for regularization.
- `--project-name`: (Optional, default: "lipid") W&B project name.
- `--experiment-name`: (Optional, default: auto-generated timestamp) Custom experiment name.
- `--accelerator`: (Optional, default: "gpu") Hardware accelerator to use (cpu/gpu/tpu).
- `--devices`: (Optional, default: -1) Number of devices to use (-1 for all available).
- `--precision`: (Optional, default: 32) Training precision (16/32).
- `--save-top-k`: (Optional, default: 3) Number of best checkpoints to save.
- `--no-pretrained`: (Optional, default: false) Use randomly initialized backbone instead of pretrained.
- `--no-normalize`: (Optional, default: false) Disable embedding normalization.

Model weights will be saved in the `data/models/contrastive_net` directory. If you want to use your own model for further steps, manually rename and move the best checkpoint to `data/models/contrastive_net/baseline.ckpt`.

#### 1.7.3 Generating Sky Finder Descriptors

To generate the descriptors for the Sky Finder dataset, execute the following commands:

```bash
cd src/pipeline
python generate_sky_finder_descriptors.py [-w <workers>] [-f]
```

**Parameters:**
- `-w`, `--n-workers`: (Optional, default: 1) Number of workers for data loading.
- `-f`, `--force`: (Optional, default: false) Force overwrite existing descriptor file.

The generated descriptors will be saved in the `generated/sky_finder_descriptors.json` file.

#### 1.7.4 Plotting the SID Space

To plot the SID space and visualize the results, execute the following commands:

```bash
cd src/pipeline
python plot_sky_image_descriptor_space.py [-g <group-by-type>] [-k <n-clusters>] [-i <interactive>]
```

**Parameters:**
- `-g`, `--group-by`: (Optional, default is `sky_type`) Specifies the grouping type for the plot. Options include `sky_type` (default) and `cluster`, which groups the descriptors by their sky condition type or by clustering them into $k$ clusters, respectively.
- `-k`, `--n-clusters`: (Optional, default is 3) Specifies the number of clusters to use when grouping by cluster type. This parameter is only relevant when `--group-by` is set to `cluster`.
- `-i`, `--interactive`: (Optional, default is false) Enables interactive mode for the plot, allowing you to hover over points to see images.


#### 1.7.5 Training and Evaluating the Classification Head

To train the classification head and evaluate the downstream classification performance:

```bash
cd src/sky_class_net
python sky_class_train.py [-e <epochs>] [-b <batch-size>] [-w <workers>] [--evaluation-steps <evaluation-steps>] [--learning-rate <learning-rate>] [--weight-decay <weight-decay>] [--dropout-rate <dropout-rate>] [--project-name <project-name>] [--experiment-name <experiment-name>] [--accelerator <accelerator>] [--devices <devices>] [--precision <precision>] [--save-top-k <save-top-k>]
```

**Parameters:**
- `-e`, `--epochs`: (Optional, default: 100) Number of training epochs.
- `-b`, `--batch-size`: (Optional, default: 32) Batch size for training.
- `-w`, `--n-workers`: (Optional, default: 1) Number of data loading workers.
- `--evaluation-steps`: (Optional, default: 100) Number of steps between validation runs.
- `--learning-rate`: (Optional, default: 1e-3) Learning rate for optimization.
- `--weight-decay`: (Optional, default: 1e-4) Weight decay for regularization.
- `--dropout-rate`: (Optional, default: 0.0) Dropout rate for regularization.
- `--project-name`: (Optional, default: "lipid") W&B project name.
- `--experiment-name`: (Optional, default: auto-generated timestamp) Custom experiment name.
- `--accelerator`: (Optional, default: "gpu") Hardware accelerator to use (cpu/gpu/tpu).
- `--devices`: (Optional, default: -1) Number of devices to use (-1 for all available).
- `--precision`: (Optional, default: 32) Training precision (16/32).
- `--save-top-k`: (Optional, default: 3) Number of best checkpoints to save.

Model weights will be saved in the `data/models/sky_class_net` directory. If you want to use your own model for further steps, manually rename and move the best checkpoint to `data/models/sky_class_net/baseline.ckpt`. To evaluate the trained classification model:

```bash
cd src/sky_class_net
python sky_class_eval.py
```

The classification results will demonstrate the effectiveness of the learned SID representations for downstream sky condition classification tasks, producing the performance metrics shown in Table 1 of Section 1.5.2.

#### 1.7.6 Plotting New Data in the SID Space

To project new sky videos into the SID space, follow these steps:

1. **Prepare the new video dataset**: Ensure the new sky videos are in a compatible format (e.g., MP4, AVI, MOV, MKV) and stored in the [data/videos/processed](data/videos/processed) directory. The videos should contain visible sky regions for accurate descriptor extraction.

2. **Preparte the manually annotated masks**: If you have manually annotated masks for the new videos, place them in the [data/videos/masks](data/videos/masks) directory. This step is optional, typically used for datasets where specific regions of interest need to be focused on.

3. **Run the projection script**: Execute the following command to process the new videos and project them into the SID space:

    ```bash
    cd src/pipeline
    python run_pipeline [-vp <video-path>] [-mp <mask-path>] [-fr <frame-rate>] [-w <workers>] [-sam2 <sam2-type>] [-gdino <gdino-type>] [-bt <box-threshold>] [-tt <text-threshold>] [-sp] [-f]
    ```

    **Parameters:**
    - `-vp`, `--video-path`: Path to the video file.
    - `-mp`, `--mask-path`: (Optional) Path to the manually annotated mask file. If provided, the script will use this mask to focus on specific regions of interest.
    - `-fr`, `--frame-rate`: (Optional, default: 1/3) Frame rate for processing the video. Higher values will extract more frames but require more processing time.
    - `-sam2`, `--sam2-type`: (Optional, default: "large") Type of SAM2 model to use for segmentation. Options include "large", "medium", and "base" or "small".
    - `-gdino`, `--gdino-type`: (Optional, default: "tiny") Type of G-DINO model to use for segmentation. Options include "tiny" or "base".
    - `-bt`, `--box-threshold`: (Optional, default: 0.35) Box threshold for SAM2 segmentation.
    - `-tt`, `--text-threshold`: (Optional, default: 0.35) Text threshold for SAM2 segmentation.
    - `-sp`, `--show-plots`: (Optional, default: false) If set, displays the generated plots for the projected SID space.
    - `-f`, `--force`: (Optional, default: false) Forces reprocessing of the video even if the descriptors already exist.

4. **Plot the SID space**: After processing the new videos, you can visualize the projected descriptors in the SID space by executing:

    ```bash
    cd src/pipeline
    python plot_pipeline.py [-vp <video-path>] [-pt]
    ```

    **Parameters:**
    - `-vp`, `--video-path`: Path to the video file.
    - `-pt`, `--plot-time`: (Optional, default: false) If set, plots the descriptors over time, showing how the SID space evolves throughout the video.

    Or simply run the following command to plot all the generated descriptors in the SID space:

    ```bash
    cd src/pipeline
    python plot_pipeline_all.py [-p <pipeline-path>]
    ```

    **Parameters:**
    - `-p`, `--pipeline-path`: (Optional, default: [generated/pipeline](generated/pipeline)) Path to the directory containing the generated descriptors.



## 2. Cloud Coverage

The cloud coverage descriptor provides a quantitative measure of sky conditions by estimating the percentage of sky pixels covered by clouds. This descriptor leverages deep learning-based segmentation to distinguish between clear sky and cloud regions, outputting a continuous value between 0 (completely clear) and 1 (completely overcast). Unlike categorical classification approaches, this regression-based method captures the nuanced gradations in cloud coverage that characterize real-world sky conditions.



### 2.1 Datasets

#### 2.1.1 Sky Finder Cover Dataset

In this repository, we introduce the Sky Finder Cover Dataset, which is a manually annotated subset of the Sky Finder Dataset with pixel-level cloud segmentation masks. This carefully curated dataset maintains the same classification schema (clear, partial, and overcast) as the original Sky Finder Dataset, providing high-quality ground truth for cloud segmentation tasks.

The dataset was created through a meticulous annotation process:
- **Selection:** Representative images were selected from each sky condition category to ensure diversity across weather conditions, times of day, and cloud formations.
- **Manual Segmentation:** Annotators created pixel-precise masks, where each pixel is labeled as either cloud-covered (white), partially covered (gray) or clear sky/ground (black). Special attention was given to cloud boundaries and transitional regions to ensure accurate coverage estimation.

For experimental evaluation, the dataset was divided into training and validation sets containing 182 and 58 images, respectively, maintaining representative distributions across all sky condition classes.

#### 2.1.2 Sky Finder Active Dataset

To address the limited size of manually annotated data, we implement an active learning framework that leverages high-confidence pseudo-labels from the broader Sky Finder dataset:

- **Initial Model Training:** A cloud coverage model was first trained on the manually annotated Sky Finder Cover Dataset using the architecture and training procedure described in Sections 2.2 and 2.4.
- **Pseudo-Label Generation:** The trained model was systematically applied to unlabeled images from the full Sky Finder Dataset, where prediction uncertainty was quantified using pixel-wise entropy measurements across the segmentation output. Through this uncertainty quantification process, only good predictions exhibiting low entropy were selected as pseudo-labels, ensuring quality control through confidence-based filtering. This threshold-based selection mechanism effectively retained only the most confident predictions for training augmentation, maintaining annotation quality while significantly expanding the available training data.

This active learning approach expands the training set with 359 high-confidence pseudo-labeled images and the validation set with 128 additional pseudo-labeled images, significantly increasing the available training data while maintaining annotation quality through automated confidence filtering.



### 2.2 Model Architecture

The cloud coverage descriptor employs a U-Net [8] architecture with a ResNet50 backbone pretrained on ImageNet serving as the feature encoder. This encoder-decoder structure is specifically designed for dense prediction tasks, making it well-suited for pixel-level cloud segmentation.

**Encoder (ResNet50 Backbone):** The ResNet50 encoder progressively downsamples input images while extracting hierarchical features at multiple scales. The pretrained weights provide robust low-level feature representations that transfer effectively to sky imagery, capturing edges, textures, and structural patterns essential for cloud boundary detection.

**Decoder with Skip Connections:** The decoder consists of upsampling blocks that progressively restore spatial resolution through bilinear interpolation followed by convolutional layers. Skip connections from corresponding encoder levels are concatenated with decoder features at each resolution level, preserving fine-grained spatial information essential for accurate cloud boundary delineation.

**Dual-Output Design:** The architecture incorporates two complementary outputs to enhance learning:
1. **Primary Segmentation Output:** Pixel-wise cloud coverage estimation through the standard U-Net segmentation head, producing a probability map where each pixel represents the likelihood of cloud coverage.

2. **Auxiliary Classification Branch:** A secondary convolutional branch processes feature maps before the final segmentation layer to output a single scalar value between 0 and 1, representing overall sky condition intensity (0 for clear, 0.5 for partial, 1 for overcast).

This auxiliary branch provides additional supervisory signal during training, enables evaluation of global sky classification accuracy, and enforces consistency between pixel-level predictions and image-level sky conditions, resulting in more robust and interpretable cloud coverage estimates.



### 2.3 Training Objective

The training objective combines three complementary loss functions to optimize both segmentation accuracy and classification consistency:

$$\mathcal{L} = 0.5 \cdot \mathcal{L}\_{F} + 0.5 \cdot \mathcal{L}\_{D} + 0.1 \cdot \mathcal{L}\_{B}$$

The focal loss ($\mathcal{L}_{F}$) addresses class imbalance and focuses learning on difficult examples:

$$\mathcal{L}_{F} = -\alpha(1-p_t)^\gamma\log(p_t)$$

Where $p_t$ is the predicted probability for the true class, $\alpha=0.5$ balances class importance, and $\gamma=2.0$ down-weights easy examples, forcing the model to focus on challenging cloud boundaries and ambiguous regions.

The dice loss ($\mathcal{L}_{D}$) optimizes spatial overlap between predicted and ground truth segmentations:

$$\mathcal{L}\_{D} = 1 - \frac{2\sum\_{i}^{N}p_i g_i}{\sum\_{i}^{N}p_i^2 + \sum\_{i}^{N}g_i^2 + \epsilon}$$

Where $p_i$ and $g_i$ are predicted and ground truth probabilities for pixel $i$, $N$ is the total number of pixels, and $\epsilon$ ensures numerical stability. This loss is particularly effective for segmentation tasks as it directly optimizes the overlap metric used for evaluation.

For the auxiliary classification branch, binary cross-entropy loss ($\mathcal{L}_{\text{BCE}}$) provides supervision using image-level sky condition labels:

$$\mathcal{L}_{B} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

Where $y$ is the ground truth sky condition class and $\hat{y}$ is the predicted classification score. This ensures consistency between pixel-level and image-level predictions.



### 2.4 Training Procedure

The cloud coverage model was trained using a two-stage approach to leverage both manually annotated and pseudo-labeled data effectively:

#### 2.4.1 Manual Labels Only:

- **Optimizer:** AdamW with learning rate $10^{-4}$ and weight decay $10^{-4}$.
- **Batch Size:** 2 images per batch.
- **Training Duration:** 100 epochs.
- **Learning Rate Scheduler:** Reduce on plateau with patience of 1 epoch and factor of 0.5.

#### 2.4.2 Active Learning Enhancement:

- **Initialization:** Best checkpoint from Stage 1.
- **Additional Data:** 359 pseudo-labeled training images, 128 pseudo-labeled validation images.
- **Optimizer:** AdamW with learning rate $10^{-4}$ and weight decay $10^{-4}$.
- **Batch Size:** 2 images per batch.
- **Training Duration:** 50 additional epochs with early stopping based on validation loss.
- **Learning Rate Scheduler:** Reduce on plateau with patience of 1 epoch and factor of 0.5.

**Hardware Configuration:** Training was conducted on a single NVIDIA RTX 3080 GPU with 10GB memory, enabling efficient processing of high-resolution sky images while maintaining reasonable training times.



### 2.5 Results

#### 2.5.1 Quantitative Performance Analysis

The cloud coverage model demonstrates strong performance across multiple evaluation metrics, with the active learning approach showing consistent improvements over the baseline model trained solely on manual annotations. The active learning enhanced model demonstrates superior performance across all metrics, with IoU improvements ranging from 2.7% to 7.1% depending on the validation set composition. The best configuration achieves a mean absolute coverage error of 8.24%, representing good accuracy in quantitative cloud coverage estimation across diverse sky conditions.

<div align="center">
   <em><strong>Table 2:</strong> Comprehensive performance comparison across training and validation configurations. Coverage Error represents the mean absolute percentage error in estimating cloud coverage. Sky Class Error represents the classification error rate for three-class sky condition categorization.</em>
</div>

<div align="center">

| Training Data | Validation Data | IoU | Dice Score | Coverage Error | Sky Class Error |
|---------------|-----------------|-----|------------|----------------|-----------------|
| Manual Labels Only | Manual Validation | 0.3632 | 0.4605 | 0.1380 | 0.2472 |
| Manual Labels Only | Manual + Pseudo Validation | 0.3697 | 0.4665 | 0.0927 | 0.2840 |
| Manual + Pseudo Labels | Manual Validation | 0.3905 | 0.4825 | 0.1365 | 0.2107 |
| Manual + Pseudo Labels | Manual + Pseudo Validation | **0.4408** | **0.5217** | **0.0824** | **0.2114** |

</div>

#### 2.5.2 SID Space Visualization with Cloud Coverage

To understand the relationship between learned sky representations and cloud coverage estimates, we visualized the Sky Image Descriptor (SID) space colored by predicted cloud coverage values. This analysis reveals important insights about model performance and limitations.

<div align="center">
  <img src="generated/sky_image_descriptor_space/sky_image_descriptor_space_cloud_cover.png" alt="SID Space with Cloud Coverage" width="90%">
  <br>
  <em><strong>Figure 4:</strong> UMAP visualization of the Sky Image Descriptor space colored by predicted cloud coverage values. Dark blue represents clear skies (low coverage), while yellow represents overcast conditions (high coverage).</em>
</div>

The visualization demonstrates clear performance patterns across different sky conditions, with the model performing exceptionally well for clear sky conditions in the rightmost cluster, consistently predicting low cloud coverage values (dark blue) that align with ground truth expectations. Moving toward more complex conditions, overcast skies with significant visual texture and structure in the upper center regions show successful high cloud coverage identification, while the central regions exhibit smooth gradations in cloud coverage estimates that effectively capture the transitional nature of partial sky conditions with appropriate intermediate values. However, a critical limitation emerges in the leftmost cluster, where uniform overcast skies with minimal texture variation display inconsistent cloud coverage predictions, showing a problematic tendency to estimate low coverage values (dark blue) despite representing heavily clouded conditions that should yield consistently high coverage estimates.

The observed performance disparity in uniform overcast conditions can be attributed to several fundamental challenges that highlight the inherent limitations of sky-only analysis. During manual annotation, human annotators naturally incorporated contextual cues from ground regions to assess sky conditions, utilizing ground shadows, ambient lighting conditions, and overall scene brightness as critical indicators of atmospheric conditions that are completely unavailable when analyzing isolated sky regions. This information loss is compounded by the model's inherent texture dependency, as uniform overcast skies often lack the distinctive structural features and spatial variation patterns that the model relies upon for accurate segmentation. Furthermore, lighting ambiguity creates additional classification challenges, as uniform conditions encompass a wide spectrum of scenarios ranging from bright days with thin, diffuse cloud cover to dark, heavily clouded conditions that can appear visually similar despite representing vastly different levels of actual cloud density and coverage.



### 2.6 Reproduction Procedure



## 3. Optical Flow

































## 2. Cloud Coverage Estimation

The sky cover descriptor quantifies cloud coverage by performing regression-based segmentation of sky regions. This descriptor combines manually-labeled data from our repository with pseudo-labels derived from the Sky Finder dataset in an active learning framework. By estimating the cloud coverage percentage across all sky pixels, it provides a single numerical representation of sky conditions.



### 2.1 Datasets

#### 2.1.1 Sky Finder Cover Dataset

The Sky Finder Cover Dataset is a manually annotated subset of the Sky Finder Dataset with pixel-level cloud segmentation masks. This carefully curated dataset maintains the same classification schema (clear, partial, and overcast) as the original Sky Finder Dataset, providing high-quality ground truth for cloud segmentation tasks.

The dataset was created through a meticulous annotation process:
1. **Selection**: Representative images were selected from each sky condition category to ensure diversity.
2. **Manual Segmentation**: Annotators created pixel-precise binary masks, where each pixel is labeled as either overcast (white), partially covered (gray) or clear sky/ground (0).

For experimental evaluation, the dataset was divided into training and validation sets containing 182 and 58 images, respectively.

#### 2.1.2 Sky Finder Active Dataset

The Sky Finder Active Dataset leverages an active learning approach to expand the training data through high-confidence pseudo-labels:

1. **Initial Model Training**: A sky cover model was first trained on the manually annotated Sky Finder Cover Dataset, as detailed in Section 2.4.
2. **Pseudo-Label Generation**:
    - The trained model was applied to unlabeled images from the Sky Finder Dataset.
    - Prediction uncertainty was quantified using pixel-wise entropy measurements.
    - Only predictions with low entropy (high confidence) were selected.

For experimental evaluation, the training set was augmented with 359 high-confidence pseudo-labeled images, while the validation set was augmented with an additional 128 pseudo-labeled images. This active learning approach expands the available training data while maintaining quality through confidence-based selection.



### 2.2 Model Architecture

The sky cover descriptor employs a U-Net architecture with a ResNet50 backbone pretrained on ImageNet1K_V2 serving as the encoder. The decoder consists of upsampling blocks that progressively restore spatial resolution through bilinear interpolation, followed by convolutional layers. Skip connections from corresponding encoder levels are concatenated with decoder features at each resolution level, preserving fine-grained spatial information essential for accurate cloud segmentation.

The architecture incorporates a dual-output design to enhance learning guidance:

1. **Primary Output**: Pixel-wise cloud coverage estimation through the standard U-Net segmentation head.
2. **Auxiliary Classification Branch**: A secondary convolutional branch that processes the feature maps before the final segmentation layer to output a single scalar value between 0 and 1, representing the overall sky condition class (0 for clear, 0.5 for partial, 1 for overcast).

This auxiliary branch serves multiple purposes: it provides additional supervisory signal during training by leveraging existing image-level sky condition labels, enables evaluation of sky classification accuracy, and helps assess the quality of learned feature representations. Most importantly, it guides the learning process by enforcing consistency between pixel-level predictions and global sky conditions.



### 2.3 Training Objective

The training objective combines three complementary loss functions to optimize both segmentation accuracy and classification consistency:

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{Focal}} + 0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.1 \cdot \mathcal{L}_{\text{BCE}}$$

Where $\mathcal{L}_{\text{Focal}}$ is defined with $\alpha=0.5$ and $\gamma=2.0$ to focus on hard-to-classify examples:

$$\mathcal{L}_{\text{Focal}} = -\alpha(1-p_t)^\gamma\log(p_t)$$

Where $p_t$ is the predicted probability for the true class, $\alpha$ is the weighting factor for class balance, and $\gamma$ is the focusing parameter that down-weights easy examples.

$\mathcal{L}_{\text{Dice}}$ optimizes overlap between predicted and ground truth segmentations:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_{i}^{N}p_i g_i}{\sum_{i}^{N}p_i^2 + \sum_{i}^{N}g_i^2 + \epsilon}$$

Where $p_i$ is the predicted probability for pixel $i$, $g_i$ is the ground truth label for pixel $i$, $N$ is the total number of pixels, and $\epsilon$ is a small constant for numerical stability.

And $\mathcal{L}_{\text{BCE}}$ provides supervision for the auxiliary classification branch using binary cross-entropy:

$$\mathcal{L}_{\text{BCE}} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

Where $y$ is the ground truth sky condition class (0 for clear, 0.5 for partial, 1 for overcast) and $\hat{y}$ is the predicted classification score from the auxiliary branch.

This multi-objective loss function balances pixel-wise classification accuracy, structural similarity, and global sky condition consistency, resulting in improved cloud segmentation performance while providing interpretable classification outputs.



### 2.4 Training Procedure

Our sky cover descriptor model was trained with the following hyperparameters and configuration:

- **Optimizer**: AdamW with a learning rate of $10^{-4}$ and weight decay of $10^{-4}$.
- **Batch Configuration**: 2 batches.
- **Training Duration**: 100 epochs for the initial model trained on manual labels only, followed by 50 epochs for the active learning enhanced model with pseudo-labels.
- **Learning Rate Scheduler**: Reduce learning rate on plateau with a patience of 1 epoch and a factor of 0.5.
- **Hardware**: Single NVIDIA RTX 3080 GPU with 10GB of memory.

This configuration provides a good balance between performance and computational efficiency, allowing the model to learn meaningful cloud segmentation representations while maintaining training stability.



### 2.5 Results

The sky cover descriptor model was comprehensively evaluated using two training approaches and validation scenarios. We compare the initial model trained exclusively on manually annotated data against the active learning enhanced model that incorporates high-confidence pseudo-labels. The evaluation encompasses both manual-only validation and combined manual+pseudo-label validation datasets to assess model generalization across different data distributions.

<div align="center">
    <em>Table 1: Comprehensive performance comparison across training and validation configurations. Coverage Error represents the mean absolute percentage error in estimating cloud coverage. Sky Class Error represents the classification error rate for the three-class sky condition categorization (clear, partial, overcast).</em>
</div>

| Training Data | Validation Data | IoU | Dice Score | Coverage Error | Sky Class Error |
|---------------|-----------------|-----|------------|----------------|-----------------|
| Manual Labels Only | Manual Validation Set | 0.3632 | 0.4605 | 0.1380 | 0.2472 |
| Manual Labels Only | Manual + Pseudo Validation Set | **0.3697** | **0.4665** | **0.0927** | **0.2840** |
| Manual + Pseudo Labels | Manual Validation Set | 0.3905 | 0.4825 | 0.1365 | 0.2107 |
| Manual + Pseudo Labels | Manual + Pseudo Validation Set | **0.4408** | **0.5217** | **0.0824** | **0.2114** |

The comprehensive evaluation reveals several important insights about the active learning approach:

1. **Consistent Improvement**: The active learning enhanced model demonstrates superior performance across all metrics compared to the baseline model trained solely on manual annotations, with IoU improvements ranging from 7.5% to 19.3%.

2. **Domain Generalization**: Both models exhibit better performance when evaluated on the combined validation set that includes pseudo-labeled data, suggesting improved generalization to the broader data distribution represented in the Sky Finder dataset.

3. **Coverage Error Reduction**: The active learning approach significantly reduces coverage estimation errors, particularly evident in the combined validation scenario where coverage error decreases from 0.0927 to 0.0824.

4. **Segmentation Quality**: The Dice score improvements indicate that the active learning approach produces more accurate cloud boundary delineation, which is crucial for precise cloud coverage quantification.



### 2.6 Reproduction Procedure

#### 2.6.1 Training the Initial Sky Cover Model

To train the initial sky cover model on the manually annotated dataset, execute the following commands:

```bash
cd src/unet
python unet_train.py
```

Model weights will be saved in the [data/models/unet](data/models/unet) directory. Manually rename and move the best checkpoint to [data/models/unet/baseline_manual.ckpt](data/models/unet/baseline_manual.ckpt).

#### 2.6.2 Active Learning Enhancement

To enhance the model using the active learning approach with pseudo-labels, execute the following commands:

```bash
cd src/unet
python unet_train.py -a
```

Parameters:
- `-a`, `--active`: Enables active learning using the previously trained model checkpoint for pseudo-label generation.

The enhanced model weights will be saved in the [data/models/unet](data/models/unet) directory. Manually rename and move the best checkpoint to [data/models/unet/baseline_active.ckpt](data/models/unet/baseline_active.ckpt).

#### 2.6.3 Evaluating the Sky Cover Model

To evaluate the performance of the trained models, execute:

```bash
cd src/unet
python unet_eval.py
```
Parameters:
- `-a`, `--active`: (Optional) Use active learning checkpoint for evaluation instead of the baseline manual-only model.
- `-p`, `--with-pseudo-labelling`: (Optional) Include pseudo-labeled validation data in the evaluation.



## 3. Classification Head for Downstream Task

The classification head serves as the final component for sky condition classification, combining the texture and cover descriptors to classify images into clear, partial, and overcast scenes. This downstream task evaluates the effectiveness of our feature extraction methods in a practical sky classification scenario.

### 3.1 Dataset Preparation

To create a robust classification dataset, we manually curated a subset of the Sky Finder dataset with the following preprocessing steps:

1. **Manual Labeling**: We manually labeled a comprehensive subset of Sky Finder images, ensuring accurate ground truth for the three sky conditions.
2. **Night Sky Removal**: Night sky images were systematically removed from the dataset as they introduce significant lighting variations that could confound the classification task and are less relevant for most practical applications.

The final curated dataset maintains the same class distribution as reported earlier: Clear (6,335 images), Partial (6,378 images), and Overcast (8,777 images).

### 3.2 Model Architecture

The classification head employs a simple 3-layer fully connected network with dropout and ReLU activations between layers.

The network accepts either:
- **16-dimensional input**: Texture embeddings from the contrastive learning model
- **17-dimensional input**: Combined texture embeddings ($16D$) + cover prediction scalar ($1D$)
- **1-dimensional input**: Cover prediction alone for baseline comparison

### 3.3 Experimental Results

Our comprehensive evaluation reveals significant insights about the effectiveness of different descriptor combinations:

#### 3.3.1 Performance Comparison

| Configuration | Train Accuracy | Val Accuracy | Test Accuracy | Train F1 | Val F1 | Test F1 |
|---------------|----------------|--------------|---------------|----------|--------|---------|
| **ALL** (Texture + Cover Prediction) | 0.9133 | 0.9050 | 0.9100 | 0.9100 | 0.8956 | 0.9021 |
| **CONTRASTIVE_ONLY** (Texture) | **0.9209** | **0.9069** | **0.9144** | **0.9170** | **0.8977** | **0.9052** |
| **COVER_ONLY** (Cover Prediction) | 0.7681 | 0.7923 | 0.7852 | 0.0000 | 0.0000 | 0.0000 |

#### 3.3.2 Key Findings

1. **Contrastive Learning Superiority**: The texture descriptor derived from contrastive learning demonstrates exceptional performance, achieving the highest accuracy and F1 scores across all evaluation splits. This validates our hypothesis that contrastive learning effectively captures discriminative sky condition features.

2. **Cover Prediction Limitations**: The cover prediction alone shows a critical failure mode - complete inability to classify partial sky conditions (F1 = 0.0000). The confusion matrices reveal that the model defaults to binary classification, never predicting the partial class.

3. **No Synergistic Effect**: Combining texture and cover descriptors does not improve performance over using texture features alone, suggesting that the contrastive learning approach already captures the essential characteristics needed for sky classification.

#### 3.3.3 Analysis of Cover Prediction Failure

<div align="center">
    <img src="generated/cover_prediction_ranges.png" alt="Cover prediction ranges" align="center" width="80%">
    <div align="center">
    <em>Figure 4: Distribution of cover prediction values across sky condition classes. The substantial overlap between partial and overcast classes explains the classification difficulties.</em>
    </div>
</div>

The visualization of cover prediction ranges reveals a fundamental issue: the partial and overcast classes exhibit highly overlapping value distributions. This overlap can be attributed to several factors:

1. **Texture Ambiguity**: Overcast skies often lack distinctive textures, presenting uniform gray appearances that vary primarily in brightness rather than structural patterns.
2. **Lighting Variability**: The diverse range of lighting conditions in overcast scenes creates a continuum of cloud coverage appearances that are difficult to distinguish from partially cloudy conditions.

### 3.4 Implications and Conclusions

The experimental results provide several important insights for sky condition classification:

1. **Contrastive Learning Effectiveness**: The superior performance of texture-only classification demonstrates that contrastive learning successfully learns implicit representations that encompass both textural and coverage characteristics without requiring explicit coverage quantification.

2. **Feature Redundancy**: The lack of improvement when combining descriptors suggests that the contrastive learning approach already captures the relevant information provided by the cover prediction, making the additional descriptor redundant.

3. **Practical Recommendation**: For deployment scenarios, using only the 16-dimensional texture embeddings provides the optimal balance of performance and computational efficiency.

### 3.5 Reproduction Procedure

#### 3.5.1 Generate Sky Finder Descriptors

First, extract both texture and cover descriptors for the entire dataset:

```bash
cd src/classification
python generate_sky_finder_descriptors.py
```

This will create a comprehensive descriptor file at [/generated/sky_finder_descriptors.json](/generated/sky_finder_descriptors.json) containing both texture embeddings and cover predictions for all images in the dataset.

#### 3.5.2 Train Classification Models

Train the classification head with different input configurations:

```bash
cd src/sky_class_net
python sky_class_train.py
python sky_class_train.py --contrastive-only
python sky_class_train.py --cover-only
```

The model weights will be saved in the [data/models/sky_class_net](data/models/sky_class_net) directory. Manually rename and move the best checkpoint to [data/models/sky_class_net/all_baseline.ckpt](data/models/sky_class_net/all_baseline.ckpt), [data/models/sky_class_net/contrastive_only_baseline.ckpt](data/models/sky_class_net/contrastive_only_baseline.ckpt), and [data/models/sky_class_net/cover_only_baseline.ckpt](data/models/sky_class_net/cover_only_baseline.ckpt), respectively.

#### 3.5.3 Evaluate Performance

To generate the comprehensive evaluation metrics and confusion matrices, run:

```bash
cd src/sky_class_net
python sky_class_eval.py
```

This will produce detailed performance metrics, confusion matrices, and visualizations for all model configurations, enabling direct comparison of the different approaches.

# 4. Pipeline

After these findings, we decide to only use the contrastive model and we will add other descriptors on our own initial video data. videos enable us to have temporal-based and motion-based descriptors that could improve describability of the skies.

1. cloud coverage percentage
2. Optical flow

# Reproducibility TODO

create gsam2 folder in src
cd gsam2
copy from github repository git clone https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file
remove root grounded-sam-2 to only have the gsam2 folder directly
cd checkpoints
bash download_ckpts.sh
cd gdino_checkpoints
bash download_ckpts.sh
cd ..
pip install -e .
pip install --no-build-isolation -e grounding_dino

## References

[1] Mihail et al., "Sky Finder: A Segmentation Benchmark for Sky Regions in the Wild," IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2016.

[2] He et al., "Deep Residual Learning for Image Recognition," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[3] Telea, A., "An Image Inpainting Technique Based on the Fast Marching Method," Journal of Graphics Tools, Vol. 9, No. 1, 2004.

[4] ImageNet TODO

[5] McInnes, L., Healy, J., and Melville, J., "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction," arXiv preprint arXiv:1802.03426, 2018.

[6] Cho, Y., Karmann, C., and Andersen, M., "Perception of window views in VR: Impact of display and type of motion on subjective and physiological responses," Building and Environment, Vol. 274, 2025, 112757. https://doi.org/10.1016/j.buildenv.2025.112757

[7] GSAM2 TODO

[8] UNet TODO
