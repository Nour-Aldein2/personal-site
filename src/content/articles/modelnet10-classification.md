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

## Method

## Results and Analysis

## Conclusions and Limitations

## References
Wang, Y., Sun, Y., Liu, Z., Sarma, S.E., Bronstein, M.M. and Solomon, J.M., 2019. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog), 38(5), pp.1-12.