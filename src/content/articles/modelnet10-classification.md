---
title: "3D Objects Classification"
description: "A custom implementation of DGCNN using EdgeConv from PyTorch Geometric to classify ModelNet10 and understand the capabilities of EdgeConv and DGCNN."
slug: "3d-classification-dgcnn"
publishDate: 2026-09-07
updatedDate: 2026-09-07
templateType: "case-study"
tags:
  - 3D
  - GNN
  - PointNet
  - Classification
featured: true
draft: false
heroImage: "/images/3d_classification_DGCNN/project_cover.png"
---

## Introduction
This article aims to review and reflect on `EdgeConv` operation and **DGCNN** to develop an understanding of the method and when is it suitable. In fact, while working on the new architecture `TopoLineArt` (to be published) I started working on reproducing many GNN operations. **DGCNN** is an interesting method because it eliminates the need to figure out the relationships between nodes while preparing a graph dataset. Instead, it is sufficient to start with kNN graph, and then the model will learn how nodes are connected while it's learning the task (in the case of Wang et al. (2019), the task is classification).

This case study aims to reviewing this method, and reimplementing DGCNN using PyTorch Geometric `EdgeCon` to classify the 3D objects in **ModelNet10** dataset. In particular, this article aims to reflect on the method and its applicability, while walking throw and analysing the task, and model's performance.

## Dataset
**ModelNet10** was the chosen dataset for this case study. This stems from different factors, the smaller size of the dataset compared to ModelNet40, which allows for faster training and evaluation that is sufficient for the purpose of this project. The dataset itself has 4,899 CAD objects saved in [Object File Format](https://segeval.cs.princeton.edu/public/off_format.html) which stores information about the faces, vertices, and edges of the object. Out of the box, the dataset has two splits; 3,991 for training and 908 testing. As shown in the figure on the left below, the splits are uniform per class, therefore, I resplit the data into three splits 80/5/15 for training/validation/testing (see the Figure 1, right). The stratified splitting was done to ensure that each class is represented in roughly the same proportion across the training, validation, and test sets, which helps make model training and evaluation more balanced and reliable. The dataset stistics per class before and after respliting are shown in Figures 2 and 3 respectively.

<figure>
  <div style="display: flex; gap: 1rem; width: 100%; align-items: flex-start;">
    <a
      href="../../images/3d_classification_DGCNN/class_count_2_splits.svg"
      class="figure-link"
      style="flex: 1 1 0; min-width: 0; max-width: none;"
    >
      <img
        src="../../images/3d_classification_DGCNN/class_count_2_splits.svg"
        alt="Class count using the official ModelNet10 train-test split"
        style="width: 100%; height: auto; max-width: none;"
      />
    </a>
    <a
      href="../../images/3d_classification_DGCNN/class_count_3_splits.svg"
      class="figure-link"
      style="flex: 1 1 0; min-width: 0; max-width: none;"
    >
      <img
        src="../../images/3d_classification_DGCNN/class_count_3_splits.svg"
        alt="Class count after resplitting the ModelNet10 dataset"
        style="width: 100%; height: auto; max-width: none;"
      />
    </a>
  </div>

  <figcaption>
    Figure 1: Class balance in ModelNet10: official split (left) and resplit dataset (right).
  </figcaption>
</figure>

The figures below show the distribution of each split for each class is for the veritices and faces for the shapes in the dataset. As we can see, they are quite similar for all splits, which mean that we can move to the next step and preprocess the data before developing the model. It is worth mentioning that if the distributions of the faces counts in a single class were significantly different, then it would be nessary to look into that more deeply (e.g. try to reshuffle the data such that each split has similar distribution).

<figure>
  <a href="../../images/3d_classification_DGCNN/mesh_statistics_2_splits.svg" class="figure-link">
    <img src="../../images/3d_classification_DGCNN/mesh_statistics_2_splits.svg" alt="Classes statistics" />
  </a>
  <figcaption>Figure 2: Per class and per split distribution of the dataset.</figcaption>
</figure>

### Data Preprocessing and Augmentation
Before passing the data to the model, points were sampled from the mesh faces, and each shape was normalised to fit inside a unit sphere centered at the origin. The shapes were augmented with rotation around the z-axis (n.b. rotations around other axes are possible, but will put the shapes at an awkward orientation that are unlikely to happen in the real world). Additionally, random uniform, symmetric, jitter noise was added to the vertices in the point cloud. Although additional augmentations are possible, such as random scaling or translation, this case study continue to follow the methods of augmenting the dataset to the literature being reproduced here.


## Method
Graph data, unlike image data, is unstructured. This lack of structure allows to express different sorts of relations between the nodes in the graph. This allowed for interesting developments in the field of AI for learning in graphs. These methods have proven to be very powerful when the data is represented correctly as graphs.

The success of these methods has motivated this case study (and many others I am working on, including my own research). Once again, the goal of reproducing Wang et al. (2019) is not only to reimplement their code, but to reflect on the method itself, and on its results.
### Baseline: Simplified PointNet
When experimenting with AI (or in any experiment in general), it is important to have a baseline that is good enough to compare your model (in this case, DGCNN) against; otherwise, how can you tell if your model is actually better? I find that, in the early stages of academic research and in case studies like this one, it is sufficient to implement a simplified version of existing literature. If you later find that comparing existing literature with your results is highly important, then you can implement the full version of the existing literature once you have a first version of your model working.

This baseline has three 1D convolutional layers, each activated by ReLU. The features produced by the final convolution are then aggregated across all points using max pooling. This produced a global feature vector which is then passed through a fully connected layer, followed by a final classification layer that outputs the logits. Each shape passed to the model had 2048 points and jitter noise with $\sigma = 0.02$. This model was trained using `Adam` optimiser and a learning rate of $10^{-3}$ for 1000 epochs and early stopping patience for 50 epochs to prevent overfitting.

### DGCNN with EdgeConv and PyTorch Geometric

Wang et al. (2019) use a graph representation for each shape's point cloud. Graphs are unordered data structures that are made up of nodes $\mathcal{N}$ and edges $\mathcal{E}$. The edges model the relationships between nodes, and the graph structure can be expressed as an adjacency matrix. Any function applied to graph data must have one of two important properties:

1. **Permutation Invariance (for graph-level outputs, e.g. graph classification)**: The function does not care about the order of the nodes (i.e. the rows and columns in the adjacency matrix):
$$f(PX, PAP^T) = f(X, A)$$
where $f$ is any function (e.g. a neural network), $A$ is the adjacency matrix, and $P$ is any permutation matrix.

2. **Permutation Equivariance (for node-level outputs, e.g. node classification)**: If you shuffle the input of a function in some way, its output will be shuffled in the same way; in other words, reordering the nodes causes the corresponding node-level outputs to be reordered in the same way:
$$F(PX, PAP^T) = P F(X, A)$$

<div align="center">

<svg width="900" height="390" viewBox="0 0 900 390" xmlns="http://www.w3.org/2000/svg">

  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#222"/>
    </marker>
  </defs>

  <!-- ================= PERMUTATION INVARIANCE ================= -->

  <text x="220" y="28" text-anchor="middle"
        font-family="Arial, sans-serif" font-size="22" font-weight="600">
    Permutation Invariance
  </text>

  <rect x="15" y="45" width="410" height="320" rx="12"
        fill="none" stroke="#777" stroke-width="1.5"
        stroke-dasharray="8 7"/>

  <!-- Top graph: B / A C -->
  <line x1="105" y1="85" x2="65" y2="145" stroke="#333" stroke-width="1.6"/>
  <line x1="105" y1="85" x2="145" y2="145" stroke="#333" stroke-width="1.6"/>
  <line x1="65" y1="145" x2="145" y2="145" stroke="#333" stroke-width="1.6"/>

  <circle cx="105" cy="85" r="19" fill="white" stroke="#222" stroke-width="1.6"/>
  <circle cx="65" cy="145" r="19" fill="white" stroke="#222" stroke-width="1.6"/>
  <circle cx="145" cy="145" r="19" fill="white" stroke="#222" stroke-width="1.6"/>

  <text x="105" y="86" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">B</text>
  <text x="65" y="146" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">A</text>
  <text x="145" y="146" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">C</text>

  <line x1="180" y1="116" x2="280" y2="116"
        stroke="#222" stroke-width="1.6" marker-end="url(#arrow)"/>

  <text x="230" y="100" text-anchor="middle"
        font-family="serif" font-size="21" font-style="italic">f</text>

  <text x="330" y="122" text-anchor="middle"
        font-family="serif" font-size="22" font-style="italic">λ</text>

  <!-- Bottom graph: A / C B -->
  <line x1="105" y1="220" x2="65" y2="280" stroke="#333" stroke-width="1.6"/>
  <line x1="105" y1="220" x2="145" y2="280" stroke="#333" stroke-width="1.6"/>
  <line x1="65" y1="280" x2="145" y2="280" stroke="#333" stroke-width="1.6"/>

  <circle cx="105" cy="220" r="19" fill="white" stroke="#222" stroke-width="1.6"/>
  <circle cx="65" cy="280" r="19" fill="white" stroke="#222" stroke-width="1.6"/>
  <circle cx="145" cy="280" r="19" fill="white" stroke="#222" stroke-width="1.6"/>

  <text x="105" y="221" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">A</text>
  <text x="65" y="281" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">C</text>
  <text x="145" y="281" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">B</text>

  <line x1="180" y1="251" x2="280" y2="251"
        stroke="#222" stroke-width="1.6" marker-end="url(#arrow)"/>

  <text x="230" y="235" text-anchor="middle"
        font-family="serif" font-size="21" font-style="italic">f</text>

  <text x="330" y="257" text-anchor="middle"
        font-family="serif" font-size="22" font-style="italic">λ</text>


  <!-- ================= PERMUTATION EQUIVARIANCE ================= -->

  <text x="675" y="28" text-anchor="middle"
        font-family="Arial, sans-serif" font-size="22" font-weight="600">
    Permutation Equivariance
  </text>

  <rect x="470" y="45" width="415" height="320" rx="12"
        fill="none" stroke="#777" stroke-width="1.5"
        stroke-dasharray="8 7"/>

  <!-- Top graph: A / C, B right -->
  <line x1="535" y1="80" x2="535" y2="148" stroke="#333" stroke-width="1.6"/>
  <line x1="535" y1="80" x2="600" y2="114" stroke="#333" stroke-width="1.6"/>
  <line x1="535" y1="148" x2="600" y2="114" stroke="#333" stroke-width="1.6"/>

  <circle cx="535" cy="80" r="19" fill="white" stroke="#222" stroke-width="1.6"/>
  <circle cx="535" cy="148" r="19" fill="white" stroke="#222" stroke-width="1.6"/>
  <circle cx="600" cy="114" r="19" fill="white" stroke="#222" stroke-width="1.6"/>

  <text x="535" y="81" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">A</text>
  <text x="535" y="149" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">C</text>
  <text x="600" y="115" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">B</text>

  <line x1="635" y1="114" x2="700" y2="114"
        stroke="#222" stroke-width="1.6" marker-end="url(#arrow)"/>

  <text x="668" y="98" text-anchor="middle"
        font-family="serif" font-size="21" font-style="italic">f</text>

  <!-- [alpha beta gamma]^T -->
  <rect x="735" y="62" width="72" height="105" rx="5"
        fill="white" stroke="#222" stroke-width="1.5"/>
  <line x1="735" y1="97" x2="807" y2="97" stroke="#999"/>
  <line x1="735" y1="132" x2="807" y2="132" stroke="#999"/>

  <text x="771" y="80" text-anchor="middle" dominant-baseline="middle"
        font-family="serif" font-size="20">α</text>
  <text x="771" y="115" text-anchor="middle" dominant-baseline="middle"
        font-family="serif" font-size="20">β</text>
  <text x="771" y="150" text-anchor="middle" dominant-baseline="middle"
        font-family="serif" font-size="20">γ</text>

  <!-- Bottom graph: C / A, B right -->
  <line x1="535" y1="215" x2="535" y2="283" stroke="#333" stroke-width="1.6"/>
  <line x1="535" y1="215" x2="600" y2="249" stroke="#333" stroke-width="1.6"/>
  <line x1="535" y1="283" x2="600" y2="249" stroke="#333" stroke-width="1.6"/>

  <circle cx="535" cy="215" r="19" fill="white" stroke="#222" stroke-width="1.6"/>
  <circle cx="535" cy="283" r="19" fill="white" stroke="#222" stroke-width="1.6"/>
  <circle cx="600" cy="249" r="19" fill="white" stroke="#222" stroke-width="1.6"/>

  <text x="535" y="216" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">C</text>
  <text x="535" y="284" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">A</text>
  <text x="600" y="250" text-anchor="middle" dominant-baseline="middle"
        font-family="Arial, sans-serif" font-size="16">B</text>

  <line x1="635" y1="249" x2="700" y2="249"
        stroke="#222" stroke-width="1.6" marker-end="url(#arrow)"/>

  <text x="668" y="233" text-anchor="middle"
        font-family="serif" font-size="21" font-style="italic">f</text>

  <!-- [gamma beta alpha]^T -->
  <rect x="735" y="197" width="72" height="105" rx="5"
        fill="white" stroke="#222" stroke-width="1.5"/>
  <line x1="735" y1="232" x2="807" y2="232" stroke="#999"/>
  <line x1="735" y1="267" x2="807" y2="267" stroke="#999"/>

  <text x="771" y="215" text-anchor="middle" dominant-baseline="middle"
        font-family="serif" font-size="20">γ</text>
  <text x="771" y="250" text-anchor="middle" dominant-baseline="middle"
        font-family="serif" font-size="20">β</text>
  <text x="771" y="285" text-anchor="middle" dominant-baseline="middle"
        font-family="serif" font-size="20">α</text>

</svg>

</div>

For more details about learning on graphs, feel free to refer to Hamilton (2020) or Bronstein et al. (2021). In their work, Wang et al. (2019) use the points in the point cloud as nodes, while the edges are selected to connect a node to its $k$ nearest neighbours, which allows them to use the message passing algorithm to aggregate information between nodes to update edge features: $\mathbf{e}_{ij} = h_\Theta(\mathbf{x}_i, \mathbf{x}_j)$, where $h_\Theta: \mathbb{R}^F \times \mathbb{R}^F \rightarrow \mathbb{R}^F$, with $F$ being the feature space size, $h$ being a non-linear function, and $\Theta$ being learnable parameters.

The operation to update node features is called `EdgeConv`, which is defined by applying the aggregation operation $\square$ on the edge features $\mathbf{e}_{ij}$ associated with all edges linked to a node:
$$\mathbf{x^\prime_i} = \square_{j:(i, j) \in \mathcal{E}} \mathbf{e}_{ij}$$

Since $\square$ is a symmetric aggregation function (e.g. $\max$), the order in which the neighbouring edge features $\mathbf{e}_{ij}$ are presented does not affect the updated feature $\mathbf{x}'_i$. Therefore, EdgeConv is permutation invariant to the ordering of a node's neighbours, while the EdgeConv layer as a whole is permutation equivariant with respect to the ordering of the input points. The choice of $h$ and $\square$ is very important, and in the paper the authors go over multiple choices and explain the differences between them:

**Table 1.** Comparison to existing methods. The per-point weight $w_i$ in [Atzmon et al. 2018] effectively is computed in the first layer and could be carried onward as an extra feature; we omit it for simplicity (Source: Wang et al. (2019)).

| Method     | Aggregation | Edge Function                                                                                                                   | Learnable parameters |
|------------|------------:|---------------------------------------------------------------------------------------------------------------------------------|---------------------:|
| PointNet   |         $-$ | $h_{\Theta}(\mathbf{x}_i,\mathbf{x}_j)=h_{\Theta}(\mathbf{x}_i)$                                                                |             $\Theta$ |
| PointNet++ |      $\max$ | $h_{\Theta}(\mathbf{x}_i,\mathbf{x}_j)=h_{\Theta}(\mathbf{x}_j)$                                                                |             $\Theta$ |
| MoNet      |      $\sum$ | $h_{\theta_m,w_n}(\mathbf{x}_i,\mathbf{x}_j)=\theta_m\cdot\left(\mathbf{x}_j\odot g_{w_n}(u(\mathbf{x}_i,\mathbf{x}_j))\right)$ |       $w_n,\theta_m$ |
| PCNN       |      $\sum$ | $h_{\theta_m}(\mathbf{x}_i,\mathbf{x}_j)=(\theta_m\cdot\mathbf{x}_j)g(u(\mathbf{x}_i,\mathbf{x}_j))$                            |           $\theta_m$ |

In EdgeConv, Wang et al.'s (2019) choice was:
$$h_\Theta (\mathbf{x}_i, \mathbf{x}_j) = \bar{h}_\Theta (\mathbf{x}_i, \mathbf{x}_j - \mathbf{x}_i)$$
where $\mathbf{x}_i$ provides the global shape structure of a patch centered at $\mathbf{x}_i$, and $\mathbf{x}_j - \mathbf{x}_i$ captures the local neighbourhood information (notice how PointNet and other methods are special cases of the EdgeConv operation). 

Based on their experiments, the authors suggest that reconstructing the graph is essential for improved results; **this is the dynamic part of their convolution operation**. In particular, they do not reconstruct the entire graph; they only rewire the nodes by recomputing the adjacency matrix each time the nodes get updated (i.e. "recompute the graph using nearest neighbors in the feature space produced by each layer"). This was done by:
1. computing a pairwise distance matrix in the feature space,
2. then taking the closest $k$ points to each point.

It is worth highlighting that this entire operation is *permutation* and *partial translation* invariant. In this way, the model not only learns how to extract local geometric features and how to group points in a point cloud; therefore, distances in deeper layers carry semantic information over long distances in the original embedding.

In this case study, we conduct three experiments:
1. **Exp. 1**: 512 nodes, with 10 kNN
2. **Exp. 2**: 1024 nodes, with 20 kNN
3. **Exp. 3**: 2048 nodes, with 20 kNN

The code used for these experiments is available in this [GitHub repository](https://github.com/Nour-Aldein2/Point-Cloud-Classification/tree/main/DGCNN). All DGCNN models were trained with jitter noise of $\sigma=0.02$ and optimized using SGD with a momentum of $0.9$ and weight decay of $10^{-4}$. The learning rate was decreased from $0.01$ to $0.0001$ over the course of training. Each model was trained for up to 250 epochs, with early stopping applied using a patience of 50 epochs. The Leaky ReLU slope was set to $0.2$, the dropout rate to $0.5$, and the batch-normalization momentum to $0.1$. With the exception of early stopping, these settings closely follow those used by Wang et al. (2019). The training histories for these three experiments are shown below:
<figure style="margin: 0; text-align: center;">
  <div style="display: flex; justify-content: center; align-items: flex-start; gap: 12px;">
    <a href="../../images/3d_classification_DGCNN/exp_512_10_loss_accuracy.svg" class="figure-link" style="width: 32%;">
      <img src="../../images/3d_classification_DGCNN/exp_512_10_loss_accuracy.svg" alt="Loss and accuracy curves for 512 points and k=10" style="width: 100%; height: auto;" />
    </a>
    <a href="../../images/3d_classification_DGCNN/exp_1024_20_loss_accuracy.svg" class="figure-link" style="width: 32%;">
      <img src="../../images/3d_classification_DGCNN/exp_1024_20_loss_accuracy.svg" alt="Loss and accuracy curves for 1024 points and k=20" style="width: 100%; height: auto;" />
    </a>
    <a href="../../images/3d_classification_DGCNN/exp_2048_20_loss_accuracy.svg" class="figure-link" style="width: 32%;">
      <img src="../../images/3d_classification_DGCNN/exp_2048_20_loss_accuracy.svg" alt="Loss and accuracy curves for 2048 points and k=20" style="width: 100%; height: auto;" />
    </a>
  </div>
  <figcaption style="margin-top: 10px;">
    <strong>Figure 3:</strong> Training loss and accuracy curves for DGCNN experiments using 512 points with $k=10$ (left), 1024 points with $k=20$ (middle), and 2048 points with $k=20$ (right).
  </figcaption>
</figure>

## Results and Analysis
In the Baseline and DGCNN experiments, the models was evaluated on the same held-out test split. Here, the quantitative results are presented, followed by the qualitative results, finally the learned embeddings in the latent space are visualised using t-SNE.

### Quantitative

The table below shows the classification performance on the ModelNet10 test set (743 samples). The baseline model is compared against DGCNN models evaluated at three input resolutions (512 points with 10 edges, 1024 points with 20 edges, and 2048 points with 20 edges). Mean class accuracy, precision, and F1-score are macro-averaged over the 10 classes. The Exp. 1 with 512-10 point-edge configuration achieves the best performance across all metrics. We also notice that the performance drops across all metrics as we increase the number of points, this alongside the results in Table 5 in Wang et al. (2019) suggest that selecting the number of nodes and edges/k neighbours plays a crucial role in DGCNN performance, and thus a hyperparameters sweep is needed before deciding the best model.

| Model                | Overall Accuracy | Mean Class Accuracy (Recall) | Precision | F1-score |
|----------------------|------------------|------------------------------|-----------|----------|
| **Baseline**         | 0.92             | 0.89                         | 0.90      | 0.89     |
| **Exp. 1: 512, 10**  | **0.96**         | **0.94**                     | **0.95**  | **0.94** |
| **Exp. 2: 1024, 20** | 0.95             | 0.92                         | 0.94      | 0.93     |
| **Exp. 3: 2048, 20** | 0.91             | 0.90                         | 0.89      | 0.89     |


Figure 5 compares the overall test accuracy of the four configurations. The 512-point model with $k=10$ achieves the highest accuracy of $0.96$, followed by the 1024-point model with $k=20$ at $0.95$. The baseline achieves an accuracy of $0.92$, while the 2048-point model with $k=20$ achieves the lowest accuracy of $0.91$. These results show that increasing the number of input points does not necessarily improve classification performance. 

<figure style="margin: 0; text-align: center;">
  <a href="../../images/3d_classification_DGCNN/overall_accuracy.png" class="figure-link">
    <img src="../../images/3d_classification_DGCNN/overall_accuracy.png" alt="Overall test accuracy across DGCNN configurations" style="width: 85%; height: auto;" />
  </a>
  <figcaption style="margin-top: 10px;"><strong>Figure 5:</strong> Overall test accuracy for the baseline and DGCNN models using 512 points with $k=10$, 1024 points with $k=20$, and 2048 points with $k=20$.</figcaption>
</figure>

Figure 6 provides a class-level comparison of precision, recall, and F1-score. The 512-point model with $k=10$ generally achieves the most consistent performance across the ten classes, with particularly strong results for Bathtub, Bed, Chair, Monitor, Sofa, and Toilet. The largest differences between configurations appear for the more challenging classes, particularly Desk, Dresser, and Night Stand. For example, the 2048-point model achieves a recall of $0.66$ for Desk and a precision of $0.66$ for Dresser. This suggests that the differences in overall performance are largely driven by a small number of difficult classes rather than by a uniform change across all classes.
<figure style="margin: 0; text-align: center;">
  <a href="../../images/3d_classification_DGCNN/per_class_metrics.png" class="figure-link">
    <img src="../../images/3d_classification_DGCNN/per_class_metrics.png" alt="Per-class precision, recall, and F1-score across DGCNN configurations" style="width: 100%; height: auto;" />
  </a>
  <figcaption style="margin-top: 10px;"><strong>Figure 6:</strong> Per-class precision, recall, and F1-score for the baseline and DGCNN models using 512 points with $k=10$, 1024 points with $k=20$, and 2048 points with $k=20$.</figcaption>
</figure>

### Qualitative
<figure style="margin: 0; text-align: center;">
  <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px;">
    <a href="../../images/3d_classification_DGCNN/confusion_matrix_baseline.png" class="figure-link">
      <img src="../../images/3d_classification_DGCNN/confusion_matrix_baseline.png" alt="Confusion matrix for the baseline DGCNN experiment" style="width: 100%; height: auto;" />
    </a>
    <a href="../../images/3d_classification_DGCNN/confusion_matrix_512.png" class="figure-link">
      <img src="../../images/3d_classification_DGCNN/confusion_matrix_512.png" alt="Confusion matrix for the DGCNN experiment with 512 points" style="width: 100%; height: auto;" />
    </a>
    <a href="../../images/3d_classification_DGCNN/confusion_matrix_1024.png" class="figure-link">
      <img src="../../images/3d_classification_DGCNN/confusion_matrix_1024.png" alt="Confusion matrix for the DGCNN experiment with 1024 points" style="width: 100%; height: auto;" />
    </a>
    <a href="../../images/3d_classification_DGCNN/confusion_matrix_2048.png" class="figure-link">
      <img src="../../images/3d_classification_DGCNN/confusion_matrix_2048.png" alt="Confusion matrix for the DGCNN experiment with 2048 points" style="width: 100%; height: auto;" />
    </a>
  </div>
  <figcaption style="margin-top: 10px;">
    <strong>Figure 4:</strong> Normalized confusion matrices for the baseline DGCNN model (top left) and experiments using 512 (top right), 1024 (bottom left), and 2048 (bottom right) points per point cloud. Bubble size and color intensity indicate the recall for each true–predicted class pair.
  </figcaption>
</figure>

### Analysis

## Conclusions and Limitations

Previously I highlighted this point:
>It is worth highlighting that this entire operation is *permutation* and *partial translation* invariant. In this way, the model not only learns how to extract local geometric features and how to group points in a point cloud; therefore, distances in deeper layers carry semantic information over long distances in the original embedding.

I believe such a technique could have important applications in astrophysics and cosmology, particularly if it were paired with a physics objective (e.g. a loss function). It could help characterize the relationships between stars in a stellar stream (e.g. [GD-1](https://en.wikipedia.org/wiki/GD-1)) in a high-dimensional feature space and identify perturbations in the stream. This could, in turn, help determine whether observed structures such as gaps and spurs are consistent with interactions with dark-matter subhalos.

## References
Bronstein, M.M., Bruna, J., Cohen, T. and Veličković, P., 2021. Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478.

Hamilton, W.L., 2020. Graph Representation Learning. Synthesis Lectures on Artificial Intelligence and Machine Learning, 14(3), pp.1–159.

Wang, Y., Sun, Y., Liu, Z., Sarma, S.E., Bronstein, M.M. and Solomon, J.M., 2019. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog), 38(5), pp.1-12.

