## TODOs

**Decoupled training**
- [x] The two stages of decoupled training's basic idea and intuition, see [[#The Basics of Decoupled Training]]
- [x] Justify the usage of decoupled training (compared with re-sampling and re-weighting), see [[#Motivation and Improvement Over Other LTL Methods]]
- [x] the tail class-aware spherical sampler, see [[#Generalizing to 3D Point Clouds (ResPointNet-LT)]]
- Core hyperparameters: 
    - Learning rate, e.g., 1/10 or 1/100 of the learning rate in Stage 1
    - Data augmentation, e.g., 1) how to balance the sampling, 2) rotation, jittering, scaling, point dropout
    - Which layers to freeze, e.g., unfreeze last two layers of the decoder
    - Classifier variants (cRT vs. LWS)

## Decoupling training
### The Basics of Decoupled Training 🤔
#### Basic Idea & Intuition

The core idea of decoupled training is to **separate representation learning from classifier learning**. Instead of trying to do both perfectly in one step, we break the process into two distinct stages:
- Stage 1: Representation Learning: Train the entire network (encoder + classifier/decoder) on the original, imbalanced dataset.
- Stage 2: Classifier Learning: Freeze the feature extractor (encoder) and re-train only the classifier/decoder on **a class-balanced dataset**. WHy on a class-balanced dataset? Because the features learned from head classes are still incredibly valuable for the tail classes.

The **intuition** can be understood with a simple vivid analogy:
Imagine a student preparing for an exam with many easy questions (head classes) and a few very tricky, rare questions (tail classes).
- One-Stage Training (Standard): The student studies all materials at once. They spend most of their time on the easy topics because they appear most often, and they never truly master the tricky ones. Their final knowledge is biased towards the easy topics.
- Decoupled Training:
    - Stage 1 (Representation): The student first reads all the textbooks, including the common ones. This builds a strong, broad foundation of knowledge (a good feature representation). Even the simple topics teach fundamental concepts.
    - Stage 2 (Classification): The student then takes the foundational knowledge and does targeted practice drills only on the exam questions, with **an equal number of easy and tricky questions**. This fine-tunes their decision-making process without damaging their core knowledge.

#### Motivation and Improvement Over Other LTL Methods

Decoupled training was motivated by a key insight: the features learned from head classes are still incredibly valuable for the tail classes. A pipe (head class) teaches a model about cylindrical shapes, metallic textures, and linear structures—knowledge that is directly useful for recognizing a pump (tail class).

Here's why it's often better than other LTL techniques:
```
vs. Re-sampling:
Undersampling throws away valuable data from head classes, weakening the feature extractor. Oversampling repeatedly shows the model the same few tail-class examples, leading to severe overfitting. **Decoupling uses all data to learn robust features and only uses balanced sampling for the lightweight classifier tuning**, avoiding both pitfalls.

vs. Re-weighting (e.g., Class-Balanced Loss):
Re-weighting creates an optimization conflict. The network is **simultaneously pushed to learn good general features while also being heavily penalized for misclassifying rare classes**. This can lead to distorted gradients and a "seesaw" effect, where improving tail classes hurts head classes significantly.
Decoupling resolves this conflict. Stage 1 focuses purely on good representations. Stage 2 focuses purely on an unbiased decision boundary. The objectives are separated, leading to a more stable and effective solution.
```

#### Why Decoupling is Crucial for 3D Point Clouds
Beyond the general benefits, decoupling is particularly powerful for 3D point cloud segmentation due to the unique nature of geometric data.

-   **Transferability of Geometric Primitives:** Head classes like pipes and beams provide millions of points that teach the encoder to be an expert at recognizing fundamental geometric primitives (e.g., cylinders, planes, edges) from dense data. These primitives are the building blocks of tail classes (a pump is an assembly of cylinders, etc.). Decoupling is crucial because Stage 1 allows the network to master these primitives without distortion from aggressive re-weighting.
-   **Mitigating Gradient Interference in 3D Space:** In point clouds, a tail-class object (a valve) is often physically attached to a head-class object (a pipe). In a single-stage approach with strong re-weighting, the large, corrective gradients from the valve can "spill over" and negatively interfere with the feature learning for the adjacent pipe points. Decoupling isolates these objectives: it first learns stable representations for all objects (Stage 1) and then refines the sensitive decision boundaries between them (Stage 2).
-   **Computational Efficiency for Experimentation:** A practical benefit is that Stage 2 is significantly faster to run than a full training cycle because the heavy encoder is frozen. This allows for rapid and efficient experimentation with different sampling strategies, loss functions, and hyperparameters for the decoder.

### Implementation & Pseudo-Code 
#### 2D cls
The "classifier" is typically the final fully-connected (FC) layer. The training consists of two stages where the 1st stage is like the normal training while the 2nd stage loads the pretrained backbone with parameters frozen and only update the classifier's parameters with a small learning rate.

```python
# --- Stage 1: Representation Learning ---
model = Encoder() + Classifier() # e.g., ResNet50 backbone + FC layer
imbalanced_loader = DataLoader(imbalanced_dataset)
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS_S1):
    for images, labels in imbalanced_loader:
        outputs = model(images)
        loss = CrossEntropyLoss(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the weights of the entire network
torch.save(model.state_dict(), 'stage1_full_model.pth')

# --- Stage 2: Classifier Re-training ---
# Load the Stage 1 model
model.load_state_dict(torch.load('stage1_full_model.pth'))

# Freeze the encoder's weights
for param in model.encoder.parameters():
    param.requires_grad = False

# Create a new loader with a class-balanced sampler
balanced_loader = DataLoader(imbalanced_dataset, sampler=ClassBalancedSampler())
# Use a new optimizer for only the classifier's parameters
optimizer = Adam(model.classifier.parameters(), lr=0.0001) # Note the much smaller LR

for epoch in range(NUM_EPOCHS_S2):
    for images, labels in balanced_loader:
        outputs = model(images) # Encoder is in eval mode, classifier in train mode
        loss = CrossEntropyLoss(outputs, labels)
        loss.backward()
        optimizer.step()
```

#### 2D semseg

The "classifier" is the entire decoder network. This is similar to the code of 2D cls but updating the decoder's weights.

```python
# --- Stage 1: Representation Learning ---
model = Encoder() + Decoder() # e.g., a UNet-style architecture
imbalanced_loader = DataLoader(imbalanced_dataset)
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop is identical to classification...
# ...
torch.save(model.state_dict(), 'stage1_seg_model.pth')

# --- Stage 2: Decoder Fine-tuning ---
model.load_state_dict(torch.load('stage1_seg_model.pth'))

# Freeze the encoder's weights
for param in model.encoder.parameters():
    param.requires_grad = False

# The balanced sampler now ensures batches have a mix of pixels/patches from all classes
balanced_loader = DataLoader(imbalanced_dataset, sampler=ClassBalancedSampler())
optimizer = Adam(model.decoder.parameters(), lr=0.0001) # Optimizer for DECODER ONLY

# Training loop is identical, but only decoder weights are updated...
# ...
```

### Generalizing to 3D Point Clouds (ResPointNet-LT)

This is a direct and powerful application of the principle. Your ResPointNet++ architecture is perfectly suited for this.
- Representation Learner (Encoder): The down-sampling part of your network. This includes the initial layers and the series of Strided Residual Bottleneck (SRB) blocks that progressively reduce the number of points while increasing the feature dimensionality. This part learns hierarchical geometric features.
- Classifier (Decoder): The up-sampling part of your network. This includes the series of up-sampling layers and MLPs that propagate features from the coarser levels back to the original points, using skip connections. This part makes the final per-point semantic predictions.

Implementation Plan for ResPointNet-LT:
Stage 1: Train Full ResPointNet++.
- Goal: Learn the richest possible geometric features from your entire industrial dataset.
- Action: Train your complete ResPointNet++ model end-to-end on all the training scenes (CH, OSCG, WRT2). Use the standard Cross-Entropy loss. Let the model learn from the millions of pipe and rbeam points to understand basic shapes. Save the final model weights.

Stage 2: Fine-tune the Decoder on a Balanced Set.
- Goal: Correct the decision bias in the decoder without harming the powerful, general features learned by the encoder.
- Action:
 - Load the weights from Stage 1.
 - Freeze the Encoder: Iterate through all parameters of your encoder module (the FC, LAO, RB, and all SRB/RB blocks in the down-sampling path) and set param.requires_grad = False.
 - Create a Balanced Sampler: This is **the most critical step**. Your data loader should not sample random blocks of the point cloud anymore. Instead, it should create training batches by **preferentially sampling spheres/blocks of points that are centered on or contain instances of tail classes (pump, tank, etc.).** The Tail class-aware sampler is the key to the success of the decoupled training.

 ```markdown
The core principle of the balanced sampler is to shift from sampling random locations in the scene to sampling locations that are guaranteed to be centered on points from specific classes, with a preference for tail classes.
1. The Core Logic: From Random Centers to Class-Aware Centers
- In your baseline ResPointNet++, the training data is generated by picking a random point from the entire scene and querying a sphere or block around it. This is location-based sampling.
- For Stage 2, we will switch to the Tail class-based sampling. The process for generating each training sphere will be:
  - First, choose a semantic class (e.g., pump). We'll choose rare classes more often.
  - Then, pick a random point from that class to be the center of our sphere.
  - Query the sphere around that chosen center point.
This ensures that even rare classes are frequently the focus of our training batches.
 ```

 - Fine-tune: Train the network again with **a very small learning rate**, e.g., 1/10 or 1/100 of the learning rate in Stage 1. Because the encoder is frozen, only the weights of the decoder (the up-sampling and final FC layers) will be updated. This teaches the decoder to become an expert at identifying tail classes without forgetting how to segment the head classes.

## Advanced Innovations on Decoupled Training
Here are some novel ideas that can be built upon the standard decoupled training framework to further tackle the "dual crisis" in 3D point clouds.

### 1. Feature Space Augmentation (FSA)
- **Idea:** Instead of only augmenting the input point clouds in Stage 2, augment the data at the *feature level*. After the frozen encoder produces feature vectors for your tail-class training instances, apply techniques like SMOTE (Synthetic Minority Over-sampling Technique) directly in the feature space. This involves creating new, synthetic feature vectors by interpolating between existing ones from the same tail class.
- **Why it's innovative for 3D:** It directly tackles the "geometric sparsity" problem at the representation level. A sparse point cloud might produce a weak or ambiguous feature vector. FSA helps create a more densely populated and robust feature manifold for tail classes, making the decoder's job of learning a decision boundary much easier.

### 2. Progressive Decoder Fine-Tuning (PDFT)
- **Idea:** Your ResPointNet++ decoder has multiple up-sampling stages. Instead of fine-tuning the entire decoder at once, do it progressively. Start by unfreezing and training only the final prediction layer. Then, in a subsequent step, unfreeze and train the last up-sampling block, and so on, moving from the top of the decoder downwards.
- **Why it's innovative for 3D:** The different layers of the decoder correspond to different levels of spatial and semantic resolution. This approach allows for a more stable, coarse-to-fine adjustment of the model. It first corrects the high-level semantic predictions and then gradually refines the fine-grained spatial boundaries.

### 3. Supervised Contrastive Learning in Stage 1
- **Idea:** Enhance Stage 1 by augmenting the standard cross-entropy loss with a supervised contrastive loss. The goal is to explicitly train the encoder to produce feature representations that are more robust to the "dual crisis". It would pull feature vectors from the same class together (e.g., a sparse pump and a dense pump) while pushing them away from feature vectors of other classes.
- **Why it's innovative for 3D:** This strengthens the Stage 1 features to be inherently better at recognizing an object's essential geometric signature, regardless of how many points it's composed of. This makes the subsequent classifier fine-tuning in Stage 2 even more effective.

### other important factors
Mastering the technique requires paying attention to these key details:

- Learning Rates: This is crucial. The learning rate for Stage 2 should be significantly smaller (e.g., 1/10th or 1/100th) than in Stage 1. You are fine-tuning a sensitive decision boundary, not learning representations from scratch.
- **Aggressive** Data Augmentation in Stage 2: Augmentation is even more critical in Stage 2. Since you are repeatedly showing the model a small pool of tail-class instances, the risk of overfitting is high. Use much **stronger and more varied** augmentations here than in Stage 1 (e.g., aggressive rotation, jittering, scaling, and especially point dropout) to prevent the decoder from simply memorizing the few examples.
- Which Layers to Freeze? The standard approach is to freeze the entire encoder. However, for your paper's ablation study, you could explore unfreezing the last 1-2 layers of the encoder during Stage 2. This allows the high-level features to adapt slightly, which can sometimes provide a performance boost.
- Classifier Variants (cRT vs. LWS): The method described above is technically "fine-tuning." A strong ablation study would involve comparing it with:
    - **Classifier Re-training (cRT):** Instead of fine-tuning the decoder, you discard the Stage 1 decoder entirely and train a new, randomly initialized decoder from scratch on the frozen features. This can sometimes yield better results if the Stage 1 decoder is extremely biased.
    - **Learnable Weight Scaling (LWS):** A simpler method where you don't re-train at all. You just learn to re-scale the logit outputs of the Stage 1 model based on class frequency, which is very fast but typically less powerful than full fine-tuning.
- **Post-Hoc Model Calibration:** Models trained with class-balanced sampling can become poorly calibrated (i.e., their confidence scores don't reflect the true likelihood of being correct). After Stage 2, you can add a simple, final calibration step. Freeze the entire network and learn a single "temperature" parameter on the validation set to scale the logits before the softmax. This improves the quality of the model's confidence scores without changing its accuracy.