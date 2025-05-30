step(discretization).
step(normalization).
step(features).
step(imputation).
step(encoding).
step(rebalancing).

step(classification).


operator(discretization, kbins).
operator(discretization, binarizer).
operator(normalization, power_transformer).
operator(normalization, robust_scaler).
operator(normalization, standard).
operator(normalization, minmax).
operator(features, select_k_best).
operator(features, pca).
operator(imputation, simple_imputer).
operator(imputation, iterative_imputer).
operator(encoding, ordinal_encoder).
operator(rebalancing, near_miss).
operator(rebalancing, smote).

operator(classification, dt).
operator(classification, knn).
operator(classification, naive_bayes).


hyperparameter(kbins, n_bins, randint).
hyperparameter(kbins, encode, choice).
hyperparameter(kbins, strategy, choice).
hyperparameter(binarizer, threshold, choice).
hyperparameter(robust_scaler, with_centering, choice).
hyperparameter(robust_scaler, with_scaling, choice).
hyperparameter(standard, with_mean, choice).
hyperparameter(standard, with_std, choice).
hyperparameter(select_k_best, k, randint).
hyperparameter(pca, n_components, randint).
hyperparameter(simple_imputer, strategy, choice).
hyperparameter(iterative_imputer, initial_strategy, choice).
hyperparameter(iterative_imputer, imputation_order, choice).
hyperparameter(near_miss, n_neighbors, randint).
hyperparameter(smote, k_neighbors, randint).

hyperparameter(dt, max_depth, randint).
hyperparameter(dt, min_samples_split, randint).
hyperparameter(dt, min_samples_leaf, randint).
hyperparameter(dt, max_features, randint).
hyperparameter(dt, max_leaf_nodes, randint).
hyperparameter(dt, splitter, choice).
hyperparameter(dt, criterion, choice).
hyperparameter(knn, n_neighbors, randint).
hyperparameter(knn, weights, choice).
hyperparameter(knn, metric, choice).


domain(kbins, n_bins, [3, 7]).
domain(kbins, encode, ["ordinal"]).
domain(kbins, strategy, ["uniform", "quantile", "kmeans"]).
domain(binarizer, threshold, [0.0, 0.5, 2.0, 5.0]).
domain(robust_scaler, with_centering, [true, false]).
domain(robust_scaler, with_scaling, [true, false]).
domain(standard, with_mean, [true, false]).
domain(standard, with_std, [true, false]).
domain(select_k_best, k, [1, 4]).
domain(pca, n_components, [1, 4]).
domain(simple_imputer, strategy, ["most_frequent", "constant"]).
domain(iterative_imputer, initial_strategy, ["most_frequent", "constant"]).
domain(iterative_imputer, imputation_order, ["ascending", "descending", "roman", "arabic", "random"]).
domain(near_miss, n_neighbors,  [1, 3]).
domain(smote, k_neighbors,  [5, 7]).

domain(dt, max_depth, [1, 4]).
domain(dt, min_samples_split, [2, 5]).
domain(dt, min_samples_leaf, [1, 5]).
domain(dt, max_features, [1, 3]).
domain(dt, max_leaf_nodes, [2, 5]).
domain(dt, splitter, ["best", "random"]).
domain(dt, criterion, ["gini", "entropy"]).
domain(knn, n_neighbors, [3, 19]).
domain(knn, weights, ["uniform", "distance"]).
domain(knn, metric, ["minkowski", "euclidean", "manhattan"]).

c1 :=> mandatory([imputation, rebalancing], classification).
c3 :=> mandatory_order([imputation, normalization], classification).
c4 :=> mandatory_order([imputation, discretization], classification).
c5 :=> mandatory_order([imputation, features], classification).
c6 :=> mandatory_order([imputation, rebalancing], classification).

c7 :=> mandatory([normalization], knn).
c8 :=> mandatory([discretization], dt).
