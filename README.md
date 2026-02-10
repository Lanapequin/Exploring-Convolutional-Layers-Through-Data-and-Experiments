# Convolutional Neural Networks Analysis: Fashion-MNIST

**Student:** Laura Natalia Perilla Quintero

## Problem Description

This project investigates convolutional layers as architectural components that introduce inductive bias into neural networks. Rather than treating CNNs as black boxes, we analyze how design choices (kernel size, depth, stride, padding) affect learning performance, scalability, and interpretability.

---

## Dataset: Fashion-MNIST

### Overview

Fashion-MNIST is a dataset of Zalando's article images designed as a more challenging replacement for the original MNIST dataset.

**Properties**:
- **Training samples**: 60,000
- **Test samples**: 10,000
- **Image dimensions**: 28×28 pixels (grayscale)
- **Classes**: 10 categories of clothing items

### Class Labels

| Label | Description |
|-------|-------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### Why Fashion-MNIST for CNNs?

1. **Spatial Structure**: Clothing items have clear edges, textures, and patterns that CNNs can detect through local receptive fields
2. **Translation Invariance**: A sneaker remains a sneaker regardless of position - exactly what convolutions exploit through weight sharing
3. **Hierarchical Features**: Simple edges combine into patterns (stitching, buttons) then into parts (sleeves, collars) then complete garments - matching CNN's hierarchical architecture
4. **More Challenging than MNIST**: Requires learning actual visual features rather than simple stroke patterns, better testing convolutional inductive bias

---

## Architecture

### Baseline Model (Non-Convolutional)

**Architecture**:
```
Input (28×28×1) → Flatten (784) 
    → Dense (128, ReLU) → Dropout (0.2)
    → Dense (64, ReLU) → Dropout (0.2)
    → Dense (10, Softmax)
```

**Parameters**: ~109,000

**Purpose**: Establish reference performance when spatial structure is ignored.

**Key Limitation**: Treats all pixels as equally related, must learn spatial relationships from scratch.

---

### Convolutional Model

**Architecture**:
```
Input (28×28×1)
    → Conv2D (32 filters, 3×3, ReLU, same padding) → MaxPool (2×2)
    → Conv2D (64 filters, 3×3, ReLU, same padding) → MaxPool (2×2)
    → Flatten → Dense (128, ReLU) → Dropout (0.3)
    → Dense (10, Softmax)
```

**Parameters**: ~420,000

**Design Rationale**:

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Depth** | 2 conv layers | Fashion-MNIST is 28×28; deeper networks unnecessary |
| **Kernel Size** | 3×3 | Parameter-efficient, standard proven choice |
| **Padding** | Same | Preserves spatial dimensions, processes edges properly |
| **Pooling** | MaxPool 2×2 | Downsamples 28→14→7, provides local translation invariance |
| **Filters** | 32, 64 | Standard progression (doubling while halving spatial size) |
| **Activation** | ReLU | Faster training, reduces vanishing gradient |

**Intentional Simplicity**: Designed for interpretability and sufficiency, not maximum accuracy through excessive depth.

---

## Results

### Performance Comparison

| Model | Test Accuracy | Parameters | Training Time |
|-------|--------------|------------|---------------|
| **Baseline (FC)** | ~88.0% | 109K | Fastest |
| **CNN (3×3)** | ~90-91% | 420K | Fast |
| **CNN (5×5)** | ~89-90% | 980K | Medium |
| **CNN (7×7)** | ~88-89% | 1.8M | Slow |

### Key Findings

1. **CNN vs Baseline**: ~2-3% absolute improvement demonstrates value of convolutional inductive bias
2. **Kernel Size**: 3×3 kernels achieved best accuracy-to-parameter ratio
3. **Efficiency**: Smaller kernels with more layers beat larger kernels with fewer layers
4. **Validation**: Modern architecture principles (VGG, ResNet) validated on small dataset

### Parameter Analysis

**Baseline Model**:
- First layer: 100,480 params (784×128 + 128)
- Most parameters connect every pixel to every neuron
- Wasteful: treats distant pixels as equally related as adjacent ones

**CNN Model**:
- First conv: 320 params (3×3×1×32 + 32)
- Second conv: 18,496 params (3×3×32×64 + 64)
- Final dense: 401,536 params (3136×128 + 128)
- Most parameters in final layer (typical for CNNs)
- Convolutional layers extremely parameter-efficient

---

## Interpretation and Reasoning

### Why CNNs Outperform the Baseline

Convolutional layers exploit three fundamental properties of image data:

#### 1. Local Connectivity (Spatial Structure)

**Baseline**: Every pixel connects to every neuron - treats pixel (0,0) and pixel (27,27) as equally related to pixel (10,10)

**CNN**: Each neuron examines only a small spatial region (3×3 receptive field) - nearby pixels forming edges/textures processed together

**Impact**: CNNs naturally capture "this patch forms an edge" without learning from scratch that adjacent pixels are related

#### 2. Weight Sharing (Translation Invariance)

**Baseline**: Learning vertical edge at position (5,5) creates different weights than same edge at (10,10) - shifting image changes every weight's relevance

**CNN**: Same 3×3 kernel slides across entire image - if it learns to detect edge, detects it anywhere through parameter sharing

**Impact**: Built-in translation invariance - model doesn't need separate examples of objects in every position

#### 3. Hierarchical Feature Composition

**Baseline**: 784 pixels → hidden layer directly - must learn "sneaker" directly from raw pixels in one step

**CNN**:
- Layer 1: Edges and textures from pixels
- Layer 2: Combines edges into patterns (curves, corners)
- Dense layers: Patterns → object parts → classification

**Impact**: Architectural guidance for hierarchical learning matches how vision works

### Inductive Biases of Convolution

**Inductive bias** = assumptions built into learning algorithm about solution space

**Three Core Biases**:

1. **Locality**: Useful features are spatially localized - nearby pixels more relevant than distant ones
2. **Translation Equivariance**: Same features can appear anywhere - detection shouldn't depend on absolute position
3. **Hierarchical Composition**: Complex concepts built from simpler ones hierarchically

**Why This Matters**: These biases are informed constraints that:
- Reduce hypothesis space
- Improve sample efficiency
- Enable better generalization
- Make training faster and more stable

**Trade-off**: Only helpful when they match problem structure

### When Convolution is NOT Appropriate

| Problem Type | Why CNNs Fail | Better Alternative |
|-------------|---------------|-------------------|
| **Tabular Data** | No spatial structure; feature order arbitrary | Dense networks, XGBoost |
| **Permutation-Invariant** | Position-sensitive by design; order shouldn't matter | Set networks, Transformers |
| **Long-Range Dependencies** | Small kernels only capture local context | Transformers, RNNs |
| **Graph Data** | No regular grid structure; variable neighbors | Graph Neural Networks |
| **Irregular Time Series** | Assumes uniform sampling; breaks with gaps | Neural ODEs, interpolation |

**Principle**: Use CNNs when data has grid structure, local patterns are meaningful, same patterns appear at different positions, and hierarchical composition makes sense.

---

## SageMaker Deployment

### Deployment Strategy

Due to IAM permission restrictions on student AWS accounts, full endpoint deployment is not possible. However, all preparation steps are demonstrated:

### Local Testing Results

Local simulation confirms the model and inference pipeline work correctly:
- Model loads successfully
- Input preprocessing handles JSON requests
- Predictions match expected format
- Confidence scores are calibrated

**Production Deployment** (with permissions) would follow:
```bash
# Upload to S3
aws s3 cp model.tar.gz s3://bucket/fashion-mnist-cnn/

# Create SageMaker model
# Deploy to endpoint
# Test via API calls
```

---

## Repository Structure

```
.
├── README.md
├── data
│ ├── train-images-idx3-ubyte
│ ├── train-labels-idx1-ubyte
│ ├── t10k-images-idx3-ubyte
│ ├── t10k-labels-idx1-ubyte
│ ├── fashion-mnist_train.csv
│ └── fashion-mnist_test.csv
└── Fashion_MNIST_CNN_Analysiss.ipynb
```