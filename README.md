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

## Usage

We specify the scheme of the ML pipeline that we want to build, step by step:

```
dataset('credit-g').
metric(balanced_accuracy).
fairness_metric(demographic_parity_ratio).
performance_threshold(0.4, 0.6).
fairness_threshold(0.4, 0.6).
mining_support(0.1).

sensitive_feature(sex, [male, female]).
sensitive_feature(personal_status, ["single", "div/dep/mar", "mar/wid"]).

step(features).
step(mitigation).
step(classification).

operator(features, select_k_best).
operator(mitigation, corr_remover).
operator(classification, knn).

hyperparameter(select_k_best, k, randint).
hyperparameter(corr_remover, alpha, choice).
hyperparameter(knn, n_neighbors, randint).
hyperparameter(knn, weights, choice).
hyperparameter(knn, metric, choice).

domain(select_k_best, k, [1, 10]).
domain(corr_remover, alpha, [0.25, 0.5, 0.75, 1.0]).
domain(knn, n_neighbors, [3, 20]).
```

In the example we have:
- a Data Pre-processing step for Features Engineering;
- a Data Pre-processing step for Mitigation;
- a Modeling step for Classification (the task we want to address).

Then, we have the implementations and the hyper-parameter domains of each step:
- SelectKBest for Features Engineering, with a integer parameter k_bins that ranges from 1 to 10;
- CorrelationRemover for Mitigation, with a parameter alpha with a value in [0.25, 0.5, 0.75, 1.0];
- K-Nearest Neighbors for Classification, with an integer parameter n_neighbors that ranges from 3 to 20.

By hitting the ```Compute Graph``` button, the LogicalKB is processed to build the Problem Graph, visualized at the bottom-right corner of the GUI.

![Screenshot 2025-05-13 123222](https://github.com/user-attachments/assets/d71e22ec-9bef-4b4e-bac8-5a00c131c9f0)

When you click the **`Compute Graph`** button, HAMLET processes the **LogicalKB** and builds the **Problem Graph**, which shows up in the bottom-right corner of the interface.

Some of the nodes in the graph (arguments) represent specific parts of the search space that could be explored.

To keep things clear, the graph only shows parts of the space that are **under doubt**, meaning they're affected by some constraint.  
In the example below, you don't see any arguments for pipelines because no constraints were added to the knowledge base. Only arguments related to **sensitive groups** are shown.

Clicking the **`Run AutoML`** button starts the actual search. HAMLET explores the search space, avoiding portions that are discouraged by the current rules.

Once it's done, you can head over to the **`Data`** tab to check out all the configurations that were explored:

![Screenshot 2025-05-13 123237](https://github.com/user-attachments/assets/7dba5cfa-4e35-482d-abaf-49222a6eaee1)

The last tab, **`AutoML arguments`**, shows **recommended constraints** that HAMLET has mined from the AutoML output:

![Screenshot 2025-05-13 123300](https://github.com/user-attachments/assets/849be3aa-d731-4d36-90fe-e83f725ad834)

These suggestions can help you update the **LogicalKB**. Just review them and decide which ones you want to include.
Let’s say we decide to include the suggested constraint that requires using both **Mitigation** and **Feature Engineering** steps when applying **Knn**:

```
c1 : [] => mandatory([features, mitigation], knn).
```

After adding this constraint and clicking **`Compute Graph`** again, we get the updated graph:

![Screenshot 2025-05-13 123623](https://github.com/user-attachments/assets/8b66ee60-841f-4ed7-a25a-8e1e98fdaed2)

In this graph, edges represent attacks between arguments.
In our case argument **A0**, representing our new constraint `c1`, attacks arguments **A1**, **A2**, and **A3**, which correspond to pipeline configurations that do not satisfy the constraint (i.e., they use KNN without both Mitigation and Feature Engineering).

We can now run a new AutoML optimization, which will **exclude** these parts of the search space, focusing only on the configurations that meet the new requirement.
