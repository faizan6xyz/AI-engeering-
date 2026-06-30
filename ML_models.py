"""
================================================================================
ML MODELS REFERENCE — scikit-learn + PyTorch + Keras
================================================================================
Full constructor signatures with every argument, each commented.
This is a REFERENCE file, not meant to be run end-to-end — instantiate
whichever model you need and feed it your own data.

Organized as:
  PART 1: scikit-learn — Linear Models
  PART 2: scikit-learn — Tree-Based Models
  PART 3: scikit-learn — Ensemble Models
  PART 4: scikit-learn — Support Vector Machines
  PART 5: scikit-learn — Naive Bayes
  PART 6: scikit-learn — Nearest Neighbors
  PART 7: scikit-learn — Clustering
  PART 8: scikit-learn — Dimensionality Reduction
  PART 9: PyTorch — Building blocks + a full custom model
  PART 10: Keras / TensorFlow — Sequential + Functional API
================================================================================
"""

# ==============================================================================
# PART 1: SCIKIT-LEARN — LINEAR MODELS
# ==============================================================================
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDClassifier
)

linear_regression = LinearRegression(
    fit_intercept=True,      # whether to calculate the intercept (False if data already centered)
    copy_X=True,             # if False, X may be overwritten, saving memory
    n_jobs=None,             # number of CPU cores to use (-1 = all cores)
    positive=False,          # force coefficients to be positive
)

logistic_regression = LogisticRegression(
    penalty="l2",             # 'l1', 'l2', 'elasticnet', or None — type of regularization
    dual=False,               # dual or primal formulation (only for l2 + liblinear)
    tol=1e-4,                 # tolerance for stopping criteria
    C=1.0,                    # inverse of regularization strength (smaller = stronger reg)
    fit_intercept=True,       # whether to add a bias/intercept term
    intercept_scaling=1,      # only used when solver='liblinear' and fit_intercept=True
    class_weight=None,        # None, 'balanced', or dict — weights for imbalanced classes
    random_state=None,        # seed for reproducibility (used by some solvers)
    solver="lbfgs",           # 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'
    max_iter=100,             # max iterations for solver to converge
    # multi_class="auto",     # REMOVED in sklearn >=1.7, solver now picks automatically
    verbose=0,                # verbosity level for liblinear/lbfgs
    warm_start=False,         # reuse previous solution as init for next fit call
    n_jobs=None,              # CPU cores to use (only for multi_class='ovr')
    l1_ratio=None,            # mixing parameter for elasticnet (0=l2, 1=l1)
)

ridge = Ridge(
    alpha=1.0,                # regularization strength (higher = more regularization)
    fit_intercept=True,
    copy_X=True,
    max_iter=None,            # max iterations for conjugate gradient solver
    tol=1e-4,
    solver="auto",            # 'auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'
    positive=False,           # force coefficients positive (only with 'lbfgs' solver)
    random_state=None,
)

lasso = Lasso(
    alpha=1.0,                 # regularization strength (controls L1 penalty)
    fit_intercept=True,
    precompute=False,          # use precomputed Gram matrix to speed up calculations
    copy_X=True,
    max_iter=1000,
    tol=1e-4,
    warm_start=False,
    positive=False,
    random_state=None,
    selection="cyclic",        # 'cyclic' or 'random' — order of feature updates
)

elastic_net = ElasticNet(
    alpha=1.0,                 # overall regularization strength
    l1_ratio=0.5,               # 0=pure Ridge(L2), 1=pure Lasso(L1), mix in between
    fit_intercept=True,
    precompute=False,
    max_iter=1000,
    copy_X=True,
    tol=1e-4,
    warm_start=False,
    positive=False,
    random_state=None,
    selection="cyclic",
)

sgd_classifier = SGDClassifier(
    loss="hinge",               # 'hinge'(SVM), 'log_loss'(logreg), 'modified_huber', etc.
    penalty="l2",                # 'l2', 'l1', 'elasticnet'
    alpha=1e-4,                  # regularization multiplier
    l1_ratio=0.15,                # elasticnet mixing parameter
    fit_intercept=True,
    max_iter=1000,
    tol=1e-3,
    shuffle=True,                # shuffle training data after each epoch
    verbose=0,
    epsilon=0.1,                  # threshold for 'huber'/'epsilon_insensitive' losses
    n_jobs=None,
    random_state=None,
    learning_rate="optimal",      # 'constant','optimal','invscaling','adaptive'
    eta0=0.0,                     # initial learning rate (for 'constant'/'invscaling')
    power_t=0.5,                   # exponent for inverse scaling learning rate
    early_stopping=False,         # stop if validation score doesn't improve
    validation_fraction=0.1,       # fraction of data held out for early stopping
    n_iter_no_change=5,             # iterations with no improvement before stopping
    class_weight=None,
    warm_start=False,
    average=False,                 # compute averaged SGD weights
)


# ==============================================================================
# PART 2: SCIKIT-LEARN — TREE-BASED MODELS
# ==============================================================================
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

decision_tree_classifier = DecisionTreeClassifier(
    criterion="gini",             # 'gini', 'entropy', 'log_loss' — split quality measure
    splitter="best",               # 'best' or 'random' — strategy to choose split at each node
    max_depth=None,                # max tree depth (None = expand until pure leaves)
    min_samples_split=2,           # min samples required to split an internal node
    min_samples_leaf=1,            # min samples required to be a leaf node
    min_weight_fraction_leaf=0.0,   # min weighted fraction of total samples at a leaf
    max_features=None,             # number/fraction/'sqrt'/'log2' of features per split
    random_state=None,
    max_leaf_nodes=None,            # grow tree with this many leaves in best-first fashion
    min_impurity_decrease=0.0,       # min impurity decrease required to split
    class_weight=None,
    ccp_alpha=0.0,                   # complexity parameter for Minimal Cost-Complexity Pruning
)

decision_tree_regressor = DecisionTreeRegressor(
    criterion="squared_error",     # 'squared_error','friedman_mse','absolute_error','poisson'
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    ccp_alpha=0.0,
)


# ==============================================================================
# PART 3: SCIKIT-LEARN — ENSEMBLE MODELS
# ==============================================================================
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, AdaBoostClassifier
)

random_forest_classifier = RandomForestClassifier(
    n_estimators=100,              # number of trees in the forest
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",            # features considered per split ('sqrt','log2',int,float,None)
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,                  # whether to bootstrap-sample when building trees
    oob_score=False,                  # use out-of-bag samples to estimate generalization score
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,                 # number/fraction of samples to draw for bootstrap
)

random_forest_regressor = RandomForestRegressor(
    n_estimators=100,
    criterion="squared_error",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=1.0,                  # default differs from classifier — 1.0 = all features
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
)

gradient_boosting_classifier = GradientBoostingClassifier(
    loss="log_loss",                  # 'log_loss' or 'exponential'(AdaBoost-like)
    learning_rate=0.1,                  # shrinks contribution of each tree
    n_estimators=100,                    # number of boosting stages
    subsample=1.0,                        # fraction of samples for fitting each tree (<1 = stochastic GB)
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,                           # max depth limits tree complexity
    min_impurity_decrease=0.0,
    init=None,                              # estimator for initial predictions
    random_state=None,
    max_features=None,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    validation_fraction=0.1,
    n_iter_no_change=None,                   # enables early stopping if set
    tol=1e-4,
    ccp_alpha=0.0,
)

adaboost_classifier = AdaBoostClassifier(
    estimator=None,                # base estimator (default: DecisionTreeClassifier depth=1)
    n_estimators=50,                # max number of estimators at which boosting terminates
    learning_rate=1.0,               # shrinks contribution of each classifier
    # algorithm="SAMME",            # REMOVED in sklearn >=1.6, SAMME is now the only option
    random_state=None,
)

# Note: XGBoost / LightGBM / CatBoost are separate libraries, not part of sklearn.
# Common pattern (requires `pip install xgboost`):
#
# from xgboost import XGBClassifier
# xgb_model = XGBClassifier(
#     n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0,
#     colsample_bytree=1.0, gamma=0, reg_alpha=0, reg_lambda=1,
#     objective="binary:logistic", eval_metric=None, random_state=None,
# )


# ==============================================================================
# PART 4: SCIKIT-LEARN — SUPPORT VECTOR MACHINES
# ==============================================================================
from sklearn.svm import SVC, SVR

svc = SVC(
    C=1.0,                       # regularization strength (smaller = wider margin, more tolerant)
    kernel="rbf",                  # 'linear','poly','rbf','sigmoid','precomputed'
    degree=3,                       # degree for 'poly' kernel
    gamma="scale",                   # kernel coefficient for 'rbf','poly','sigmoid' ('scale' or 'auto')
    coef0=0.0,                        # independent term in 'poly'/'sigmoid' kernels
    shrinking=True,                    # use the shrinking heuristic
    probability=False,                  # enable probability estimates (slows down fitting)
    tol=1e-3,
    cache_size=200,                       # kernel cache size in MB
    class_weight=None,
    verbose=False,
    max_iter=-1,                           # -1 = no limit
    decision_function_shape="ovr",          # 'ovo' or 'ovr' for multiclass
    break_ties=False,
    random_state=None,
)

svr = SVR(
    kernel="rbf",
    degree=3,
    gamma="scale",
    coef0=0.0,
    tol=1e-3,
    C=1.0,
    epsilon=0.1,                  # width of the no-penalty tube around predictions
    shrinking=True,
    cache_size=200,
    verbose=False,
    max_iter=-1,
)


# ==============================================================================
# PART 5: SCIKIT-LEARN — NAIVE BAYES
# ==============================================================================
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gaussian_nb = GaussianNB(
    priors=None,                  # prior probabilities of classes (None = learned from data)
    var_smoothing=1e-9,            # portion of largest variance added for calculation stability
)

multinomial_nb = MultinomialNB(
    alpha=1.0,                    # additive (Laplace/Lidstone) smoothing parameter
    force_alpha=True,              # if False, alpha < 1e-10 is set to 1e-10
    fit_prior=True,                 # whether to learn class prior probabilities
    class_prior=None,                # custom class priors
)


# ==============================================================================
# PART 6: SCIKIT-LEARN — NEAREST NEIGHBORS
# ==============================================================================
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

knn_classifier = KNeighborsClassifier(
    n_neighbors=5,                # number of neighbors to use
    weights="uniform",              # 'uniform' or 'distance' — weight function
    algorithm="auto",                # 'auto','ball_tree','kd_tree','brute'
    leaf_size=30,                      # leaf size for BallTree/KDTree, affects speed/memory
    p=2,                                # power parameter for Minkowski metric (1=manhattan, 2=euclidean)
    metric="minkowski",                  # distance metric to use
    metric_params=None,
    n_jobs=None,
)

knn_regressor = KNeighborsRegressor(
    n_neighbors=5,
    weights="uniform",
    algorithm="auto",
    leaf_size=30,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=None,
)


# ==============================================================================
# PART 7: SCIKIT-LEARN — CLUSTERING
# ==============================================================================
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

kmeans = KMeans(
    n_clusters=8,                  # number of clusters to form
    init="k-means++",                # 'k-means++','random', or an array of initial centers
    n_init="auto",                     # number of times algorithm runs with different seeds
    max_iter=300,                        # max iterations for a single run
    tol=1e-4,                              # relative tolerance for convergence
    verbose=0,
    random_state=None,
    copy_x=True,                            # modify a copy of data, not original
    algorithm="lloyd",                       # 'lloyd' or 'elkan'
)

dbscan = DBSCAN(
    eps=0.5,                       # max distance between two samples for one to be a neighbor
    min_samples=5,                   # min samples in a neighborhood for a point to be a core point
    metric="euclidean",
    metric_params=None,
    algorithm="auto",                  # 'auto','ball_tree','kd_tree','brute'
    leaf_size=30,
    p=None,                              # power of Minkowski metric
    n_jobs=None,
)

agglomerative_clustering = AgglomerativeClustering(
    n_clusters=2,                  # number of clusters to find
    metric="euclidean",              # distance metric (affinity in older versions)
    memory=None,                       # caches output of computation tree
    connectivity=None,                  # connectivity matrix for structured clustering
    compute_full_tree="auto",
    linkage="ward",                       # 'ward','complete','average','single'
    distance_threshold=None,                # if set, n_clusters must be None
    compute_distances=False,
)


# ==============================================================================
# PART 8: SCIKIT-LEARN — DIMENSIONALITY REDUCTION
# ==============================================================================
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(
    n_components=None,            # number of components to keep (int, float for variance, 'mle')
    copy=True,
    whiten=False,                   # whiten components to have unit variance
    svd_solver="auto",                # 'auto','full','arpack','randomized','covariance_eigh'
    tol=0.0,
    iterated_power="auto",
    n_oversamples=10,
    power_iteration_normalizer="auto",
    random_state=None,
)

tsne = TSNE(
    n_components=2,               # dimension of embedded space
    perplexity=30.0,                # related to number of nearest neighbors considered
    early_exaggeration=12.0,          # controls cluster tightness early in optimization
    learning_rate="auto",
    max_iter=1000,                       # number of optimization iterations
    n_iter_without_progress=300,
    min_grad_norm=1e-7,
    metric="euclidean",
    init="pca",                            # 'pca' or 'random' initialization
    verbose=0,
    random_state=None,
    method="barnes_hut",                     # 'barnes_hut' (fast) or 'exact'
    angle=0.5,                                 # tradeoff between speed/accuracy for barnes_hut
    n_jobs=None,
)


# ==============================================================================
# PART 9: PYTORCH — building blocks + a full custom model
# ==============================================================================
# pip install torch
import torch
import torch.nn as nn

# --- Common layers with full signatures ---

linear_layer = nn.Linear(
    in_features=128,      # size of each input sample
    out_features=64,       # size of each output sample
    bias=True,               # whether to learn an additive bias
)

conv2d_layer = nn.Conv2d(
    in_channels=3,         # e.g. 3 for RGB images
    out_channels=16,         # number of filters
    kernel_size=3,             # convolution kernel size (int or tuple)
    stride=1,                    # stride of convolution
    padding=0,                     # zero-padding added to both sides
    dilation=1,                      # spacing between kernel elements
    groups=1,                          # blocked connections from input to output channels
    bias=True,
    padding_mode="zeros",                # 'zeros','reflect','replicate','circular'
)

lstm_layer = nn.LSTM(
    input_size=128,         # number of expected features in input
    hidden_size=64,           # number of features in hidden state
    num_layers=1,               # number of stacked LSTM layers
    bias=True,
    batch_first=True,             # if True, input/output shape is (batch, seq, feature)
    dropout=0.0,                    # dropout on outputs of each layer except the last
    bidirectional=False,
    proj_size=0,                       # if >0, use LSTM with projections of given size
)

dropout_layer = nn.Dropout(
    p=0.5,                # probability of an element being zeroed
    inplace=False,
)

batchnorm_layer = nn.BatchNorm1d(
    num_features=64,       # number of features/channels
    eps=1e-5,                # value added to denominator for numerical stability
    momentum=0.1,              # momentum for running_mean/running_var
    affine=True,                 # learn scale/shift parameters
    track_running_stats=True,
)

# --- A full custom model class ---
class SimpleNeuralNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleNeuralNet()

# --- Optimizer with full arguments ---
optimizer = torch.optim.Adam(
    params=model.parameters(),   # parameters to optimize
    lr=1e-3,                       # learning rate
    betas=(0.9, 0.999),              # coefficients for running averages of gradient/its square
    eps=1e-8,                          # term added for numerical stability
    weight_decay=0,                      # L2 penalty
    amsgrad=False,                         # whether to use the AMSGrad variant
)

# --- Loss function ---
criterion = nn.CrossEntropyLoss(
    weight=None,                 # manual rescaling weight for each class
    size_average=None,             # deprecated, use reduction
    ignore_index=-100,               # specifies a target value to ignore
    reduce=None,                       # deprecated, use reduction
    reduction="mean",                    # 'none','mean','sum'
    label_smoothing=0.0,                   # amount of label smoothing
)


# ==============================================================================
# PART 10: KERAS / TENSORFLOW
# ==============================================================================
# pip install tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Sequential API ---
keras_model = keras.Sequential([
    layers.Dense(
        units=128,                  # dimensionality of output space
        activation="relu",            # activation function
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        input_shape=(784,),
    ),
    layers.Dropout(
        rate=0.3,                  # fraction of input units to drop
        noise_shape=None,
        seed=None,
    ),
    layers.Dense(units=64, activation="relu"),
    layers.Dense(units=10, activation="softmax"),
])

# --- Conv2D layer (full signature) ---
conv2d_keras = layers.Conv2D(
    filters=32,                    # number of output filters
    kernel_size=(3, 3),              # size of convolution window
    strides=(1, 1),
    padding="valid",                   # 'valid' or 'same'
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation="relu",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)

# --- LSTM layer (full signature) ---
lstm_keras = layers.LSTM(
    units=64,                       # dimensionality of output space
    activation="tanh",
    recurrent_activation="sigmoid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=False,           # return full sequence vs. last output only
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
)

# --- Compile step (optimizer/loss/metrics all with key args) ---
keras_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
    ),
    loss="sparse_categorical_crossentropy",     # or keras.losses.CategoricalCrossentropy(), etc.
    metrics=["accuracy"],
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=False,
    steps_per_execution=1,
)
