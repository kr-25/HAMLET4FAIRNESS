# HAMLET: Human-centered AutoMl via Logic and argumEnTation

This is the implementation of the off-the-shelf framework **HAMLET4Fairness**, proposed in the paper: **HAMLET4Fairness: Enhancing Fairness in AI Pipelines through Human-Centered AutoML and Argumentation**.

## Requirements

- Docker
- Gradle
- Java >= 11.0

## Execution

```
./gradlew runShadow --args="[workspace_path]  [optimization_mode] [n_configurations] [time_budget] [optimization_seed]  [debug_mode] [knowledge_base_path]"
```

- **[workspace_path]**: absolute path to the file system folder containing the workspace (i.e., where to save the results); if it does not exist, a new workspace is created, otherwise, the previous run is resumed.
- **[optimization_mode]**: a string in ['min', 'max'] to specify the objective as minimization or maximization.
- **[n_configurations]**: an integer of the number of configurations to try in the optimization of each iteration.
- **[time_budget]**: the time budget in seconds given to the optimization of each iteration.
- **[optimization_seed]**: seed for reproducibility.
- **[debug_mode]**: a string in ['true', 'false'] to specify HAMLET execution in debug or release mode. In debug mode, the Docker container for the AutoML optimization is built from the local sources; otherwise the released Docker image is downloaded.
- **[knowledge_base_path]** (OPTIONAL): file system path to an HAMLET knowledge base. If provided, HAMLET is run in console (with no GUI) mode and the theory is leveraged; otherwise HAMLET GUI is launched.

## Knowledge Formalization

HAMLET allows to encode both the AutoML search space and the user-defined constraints into the LogicalKB:

**Context**:
- ```dataset(D).``` specifies the dataset ```D``` to train the AI pipelines
- ```metric(perf_metric).``` specifies the metric ```perf_metric``` to consider as the performance objective
- ```fairness_metric(fair_metric).``` specifies the metric ```fair_metric``` to consider as a fairness objective
- ```performance_threshold(min, max).``` specifies the thresholds to consider for mining discouraging (```perf_metric < min```) and promising (```perf_metric > max```) constraints
- ```fairness_threshold(min, max).``` specifies the thresholds to consider for mining unfair (```fair_metric < min```) and fair (```fair_metric > max```) constraints
- ```mining_support(supp).``` specifies the support ```supp``` to accept a mined constraint (min percentage of pipelines following the rule)

**Search Space**:
- ```step(S).``` specifies a step ```S``` of the pipeline, with ```S``` in [```discretization```, ```normalization```, ```rebalancing```, ```imputation```, ```features```, ```mitigation```, ```classification```]
- ```operator(S, O).``` specifies an operator ```O``` for the step ```S```, with ```O``` in [```kbins```, ```binarizer```, ```power_transformer```, ```robust_scaler```, ```standard```,  ```minmax```, ```select_k_best```, ```pca```, ```simple_imputer```, ```iterative_imputer```, ```near_miss```, ```smote```, , ```corr_remover```, ```lfr```, ```mlp```, ```rf```, ```knn```]
- ```hyperparameter(O, H, T).``` specifies an hyper-parameter ```H``` for the operator ```O``` with type ```T```, ```H``` can be every hyper-parameter name of the chosen Scikit-learn operator ```O```, ```T``` is chosen accordingly and has to be in [```randint```, ```choice```, ```uniform```]
- ```domain(O, H, D).``` specifies the domain ```D``` of the hyper-parameter ```H``` of the operatore ```O```, ```D``` is an array in ```[ ... ]``` brackets containing the values that the hyper-parameter ```H``` can assume (in case of ```randint``` and ```uniform```, the array has to contain just two elements: the boundary of the range)

**Constraints**:
- ```id : [] => mandatory_order([S1, S2], O1).``` specifies a ```mandatory_order``` constraint: the step ```S1``` has to appear before the step ```S2``` when occurring the operator ```O1``` of the task step (in this implementation we support only ```classification``` task); it is possible to put ```classification``` instead of ```O1```, this will apply the constraint for each ```classification``` operators
- ```id : [] => mandatory([S1, S2, ...], O1).``` specifies a ```mandatory``` constraint: the steps ```[S1, S2, ...]``` are mandatory when occurring the operator ```O1``` of the task step (in this implementation we support only ```classification``` task); if the array of the steps is empty, the constraint specifies only that O1 is mandatory (with or withour Data Pre-processing steps)
- ```id : [] => forbidden([S1, S2, ...], O1).``` specifies a ```forbidden``` constraint: the steps ```[S1, S2, ...]``` are forbidden when occurring the operator ```O1``` of the task step (in this implementation we support only ```classification``` task); if the array of the steps is empty, the constraint specifies only that O1 is forbidden (with or withour Data Pre-processing steps)

**Fairness Predicates**:
- ```sensitive_feature(F, [V1, ..., VN]).``` marks the feature `F` in the selected dataset as **sensitive**. It will be used to guide the optimization of the fairness score during pipeline evaluation or selection. The values `[V1, ..., VN]` specify the possible values of `F` to be considered. When multiple `sensitive_feature` predicates are defined (i.e., for different features), the system will construct **sensitive groups** by computing **all combinations** of the specified values across those features. These groupings serve as the basis for measuring and optimizing fairness.
- ```discriminate(pipeline([S1, S2, ...], C), O1).``` indicates that the pipeline composed of the ordered steps `[S1, S2, ...]` followed by classifier `C` results in unfair treatment of sensitive group `O1`. While the predicate does not explicitly define a `forbidden_order` constraint, it acts as one: it marks pipelines with the specified sequence as yielding low fairness metric values with respect to group `O1`. As a result, it creates a conflict with any pipeline containing the same sequence, effectively functioning like a `forbidden_order` constraint.

The two ready-to-use LogicalKB that have been used for the paper's experiments can be found in the [resources folder](https://github.com/kr-25/HAMLET4FAIRNESS/tree/db6ad0d705845bd50d18e548bf0bd81cd64dfd26/automl/resources) of this repository:
- [kb.txt](https://github.com/kr-25/HAMLET4FAIRNESS/blob/1d3e81a8216ded56d45016c4e9df2845e724569b/automl/resources/kb.txt) is a knowledge base containing the search space leveraged in our experiments;
- [pkb.txt](https://github.com/kr-25/HAMLET4FAIRNESS/blob/1d3e81a8216ded56d45016c4e9df2845e724569b/automl/resources/pkb.txt) (PreliminaryKB) is a knowledge base containing the search space along with some suggested constraints (discovered in the paper [Data pre-processing pipeline generation for AutoETL](https://www.sciencedirect.com/science/article/abs/pii/S0306437921001514)).

## Usage

<img width="960" alt="hamlet_gui" src="https://user-images.githubusercontent.com/41596745/209572069-4e63d9b5-a88d-405b-bf01-cc1a02cd1812.png">

We specify the scheme of the ML pipeline that we want to build, step by step.
In the example we have:
- a Data Pre-processing step for Discretization;
- a Data Pre-processing step for Normalization;
- a Modeling step for Classification (the task we want to address).

Then, we have the implementations and the hyper-parameter domains of each step:
- KBins for Discretization, with a integer parameter k_bins that ranges from 3 to 8;
- StandardScaler for Normalization, with no parameter;
- Decision Tree and K-Nearest Neighbors for Classification, with -- respectively -- an integer parameter max_depth that ranges from 1 to 5 and an integer parameter n_neighbors that ranges from 3 to 20.

Finally, we have a user-defined constraints (```c1```): forbid Normalization for Decision Tree.

By hitting the ```Compute Graph``` button, the LogicalKB is processed to build the Problem Graph, visualized at the bottom-right corner of the GUI.
Each node of the graph (arguments) represents a specific portion of search sub-space, the legend is visualized at the bottom-left corner.
For instance:
- A1, A3, A5, A7, and A9 represent all the possible pipelines for the Decision Tree algorithm;
- A2, A4, A6, A8, and A10 represent all the possible pipelines for the K-Nearest Neighbor algorithm.

Each constraint is represented as an argument as well: A0 represents the user-defined constraint ```c1```.

Edges are attacks from an argument to another (```c1``` attacks exactly the pipelines in which we have Normalization along with the Decision Tree).

By hitting the ```Run AutoML``` button, HAMLET triggers the exploration of the encoded search space, taking also in consideration the specified constraints (discouraging the exploration in those particular sub-spaces).

At the end of the optimization, the user can switch to the ```Data``` tab to go through all the explored configurations:

<img width="956" alt="hamlet-gui-data2" src="https://user-images.githubusercontent.com/41596745/209576316-92bb528a-b180-4b61-83fd-621a3f8e3589.png">

As to the last tab ```AutoML arguments```, we can see reccomendations of constraints, mined from the AutoML output:

<img width="959" alt="hamlet-gui-rules2" src="https://user-images.githubusercontent.com/41596745/210392351-13491f27-e07f-4e3e-a012-4f2e3692bc52.png">

The data scientist can consider the iclusion of the proposed arguments in the LogicalKB.
At this point, a new optimization can be performed.
