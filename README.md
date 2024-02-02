# UPFD-and-HGPSL
Implementation of deep Graph Neural Network with HGP-SL graph pooling layers for Fake News Detection in Twitter using Pytorch Lightning and Pytorch Geometric.
# Idea
Decided to combine the approach of collecting data proposed in UPFD with a deeper Graph Neural Network architecture as in the original paper only networks with a single graph convolution were used. Used network architecture is very similar to the on proposed in Hierarchical Graph Pooling with Structure Learning with exception of changing graph convolution operation in the first layer.
# Data
Dataset used for this experiment is the one created and published as part of User Preference-aware Fake News Detection paper. This dataset is integrated into Pytroch Geometric library and can be easily dowloaded.
It contains graphs of users that participated in spreading of fake news.
# Results
Results on Gossipcop
| Method | Acc | F1 | Prec | Rec |
| ---- | ---- | ---- | ---- | ---- |
| HGP-SL + GCN | 0.9443 | 0.9447 | 0.9405 | 0.9485 |
| HGP-SL + GAT | 0.9187 | 0.9183 | 0.9284 | 0.9118 |
| **HGP-SL + GraphSAGE** | **0.96** | **0.9604** | **0.9527** | **0.9682** |
| HGP-SL + GIN | 0.9425 | 0.9429 | 0.9376 | 0.9483 |

Results on Politifact
| Method | Acc | F1 | Prec | Rec |
| ---- | ---- | ---- | ---- | ---- |
| HGP-SL + GCN | 0.7964 | 0.7826 | 0.8617 | 0.7168 |
| HGP-SL + GAT | 0.7692 | 0.7811 | 0.7583 | 0.8053 |
| HGP-SL + GraphSAGE | 0.7963 | 0.7846 | 0.8541 | 0.7256 |
| **HGP-SL + GIN** | **0.8009** | **0.8086** | **0.7948** | **0.8230** |
# Sources
 - User Preference-aware Fake News Detection - https://arxiv.org/pdf/2104.12259.pdf
 - Hierarchical Graph Pooling with Structure Learning - https://arxiv.org/pdf/1911.05954.pdf
 - Semi-Supervised Classifiactaion with Graph Convolutional Networks - https://arxiv.org/pdf/1609.02907.pdf
 - Graph Attention Networks - https://arxiv.org/pdf/1710.10903.pdf
 - Inductive Representation Learning on Large Graphs - https://arxiv.org/pdf/1706.02216.pdf
 - How Powerful are Graph Neural Networks? - https://arxiv.org/pdf/1810.00826.pdf
