# Multi-label Punitive kNN with Self-Adjusting Memory for Drifting Data Streams

In multi-label learning, data may simultaneously belong to more than one class. When multi-label data arrives as a stream, the challenges associated with multi-label learning are joined by those of data stream mining, including the need for algorithms that are fast and flexible, able to match both the speed and evolving nature of the stream. This paper presents a punitive k nearest neighbors algorithm with a self-adjusting memory (MLSAMPkNN) for multi-label, drifting data streams. The memory adjusts in size to contain only the current concept and a novel punitive system identifies and penalizes errant data examples early, removing them from the window. By retaining and using only data that are both current and beneficial, MLSAMPkNN is able to adapt quickly and efficiently to changes within the data stream while still maintaining a low computational complexity. Additionally, the punitive removal mechanism offers increased robustness to various data-level difficulties present in data streams, such as class imbalance and noise. The experimental study compares the proposal to 24 algorithms using 30 real-world and 15 artificial multi-label data streams on six multi-label metrics, evaluation time, and memory consumption. The superior performance of the proposed method is validated through non-parametric statistical analysis, proving both high accuracy and low time complexity. MLSAMPkNN is a versatile classifier, capable of returning excellent performance in diverse stream scenarios.

# Manuscript - ACM Transactions on Knowledge Discovery from Data (TKDD)

https://dl.acm.org/doi/10.1145/3363573

# Citing Kappa Updated Ensemble

> M. Roseberry, B. Krawczyk, and A. Cano. Multi-label Punitive kNN with Self-Adjusting Memory for Drifting Data Streams. ACM Transactions on Knowledge Discovery from Data, 2019.
