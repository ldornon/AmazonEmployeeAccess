library(vroom)
library(tidymodels)
library(embed)
library(DataExplorer)
library(GGally)
library(discrim)
install.packages("discrim")
library(kknn)
install.packages("kknn")

#Read in the data
amazon_train <- vroom("./train.csv") %>% 
  mutate(ACTION = as.factor(ACTION))
amazon_test <- vroom("./test.csv")



A_Recipe <- recipe(ACTION~., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(),fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors())

prep <- prep(A_Recipe)
baked <- bake(prep, new_data = amazon_train)

glimpse(baked)



my_mod <- logistic_reg() %>% 
  set_engine("glm")

amazon_workflow <- workflow() %>% 
  add_recipe(A_Recipe) %>% 
  add_model(my_mod) %>% 
  fit(data = amazon_train)

amazon_predictions <- predict(amazon_workflow, 
                              new_data = amazon_test, 
                              type = "prob" )

log_preds <-tibble(id =amazon_test$id,
                   ACTION = amazon_predictions$.pred_1)
vroom_write(x=log_preds, file="./AmazonPreds.csv", delim=",")

#############
#Penalized Logistic Regression
#############

My_Recipe <- recipe(ACTION~., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(),fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

my_mod <- logistic_reg(mixture =tune(), penalty = tune()) %>% 
  set_engine("glmnet")

amazon_workflow <- workflow() %>% 
  add_recipe(My_Recipe) %>% 
  add_model(my_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(amazon_train, v =10 , repeats = 1)

CV_results <- amazon_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>% 
  select_best("roc_auc")

final_wf <- amazon_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)

Penalized_preds <-final_wf %>% 
  predict(new_data = amazon_test, type = "prob")


penalized_log_preds <-tibble(id =amazon_test$id,
                             ACTION = Penalized_preds$.pred_1)

vroom_write(x=penalized_log_preds, file="./AmazonPenalizedPreds.csv", delim=",")


##############
# Random Forest
#############

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")


My_Recipe <- recipe(ACTION~., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(),fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))


RF_amazon_workflow <- workflow() %>% 
  add_recipe(My_Recipe) %>% 
  add_model(my_mod)

tuning_grid <- grid_regular(mtry(range = c(1,9)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(amazon_train, v= 5, repeats = 1)


CV_results <- RF_amazon_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>% 
  select_best("roc_auc")

final_wf <-
  RF_amazon_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)

Random_forest_preds <-final_wf %>% 
  predict(new_data = amazon_test, type = "prob")

Random_preds <-tibble(id =amazon_test$id,
                             ACTION = Random_forest_preds$.pred_1)

vroom_write(x=Random_preds, file="./AmazonRFPreds.csv", delim=",")

##############

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

A_Recipe <- recipe(ACTION~., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(),fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors())


RF_amazon_workflow <- workflow() %>% 
  add_recipe(A_Recipe) %>% 
  add_model(my_mod)

tuning_grid <- grid_regular(mtry(range = c(1,9)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(amazon_train, v= 5, repeats = 1)


CV_results <- RF_amazon_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>% 
  select_best("roc_auc")

final_wf <-
  RF_amazon_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)

Random_forest_preds <-final_wf %>% 
  predict(new_data = amazon_test, type = "prob")

Random_preds <-tibble(id =amazon_test$id,
                      ACTION = Random_forest_preds$.pred_1)

vroom_write(x=Random_preds, file="./AmazonRFPreds1.csv", delim=",")

##################
# Naive Bayes

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_wf <- workflow() %>% 
  add_recipe(My_Recipe) %>% 
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness())

folds <- vfold_cv(amazon_train, v= 5, repeats = 1)


CV_results <- nb_wf %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>% 
  select_best("roc_auc")


final_wf <- nb_wf %>% finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)

NB_preds <- final_wf %>% 
  predict(nb_wf, new_data = amazon_test, type = "prob")

Naive_preds <- tibble(id =amazon_test$id,
                      ACTION = NB_preds$.pred_1)
vroom_write(x=Naive_preds, file="./AmazonNBPreds1.csv", delim=",")


##################
# K-Nearest Neighbor
##################

knn_model <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

knn_wf <- workflow() %>% 
  add_recipe(My_Recipe) %>% 
  add_model(knn_model)

tuning_grid <- grid_regular(neighbors())
folds <- vfold_cv(amazon_train, v = 5, repeats = 1)

CV_results <- knn_wf %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>% 
  select_best("roc_auc")

final_wf <- knn_wf %>% finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)

KNN_preds <- final_wf %>% 
  predict(new_data = amazon_test, type = "prob")

KNearest_preds <- tibble(id = amazon_test$id, 
                       ACTION = KNN_preds$.pred_1)
vroom_write(x = KNearest_preds, file ="./AmazonKNNPreds.csv", delim = ",")

###################
# Naive Bayes - PCA
##################


My_Recipe <- recipe(ACTION~., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(),fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = .8)
prep <- prep(My_Recipe)
baked <- bake(prep, new_data = amazon_train)

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_wf <- workflow() %>% 
  add_recipe(My_Recipe) %>% 
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness())

folds <- vfold_cv(amazon_train, v= 5, repeats = 1)


CV_results <- nb_wf %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>% 
  select_best("roc_auc")


final_wf <- nb_wf %>% finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)

NB_PCApreds <- final_wf %>% 
  predict(nb_wf, new_data = amazon_test, type = "prob")

Naive_preds <- tibble(id =amazon_test$id,
                      ACTION = NB_PCApreds$.pred_1)
vroom_write(x=Naive_preds, file="./AmazonNBPCAPreds.csv", delim=",")


###################
# KNN - PCA
##################

knn_model <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

knn_wf <- workflow() %>% 
  add_recipe(My_Recipe) %>% 
  add_model(knn_model)

tuning_grid <- grid_regular(neighbors())
folds <- vfold_cv(amazon_train, v = 5, repeats = 1)

CV_results <- knn_wf %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>% 
  select_best("roc_auc")

final_wf <- knn_wf %>% finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)

KNN_PCApreds <- final_wf %>% 
  predict(new_data = amazon_test, type = "prob")

KNearest_preds <- tibble(id = amazon_test$id, 
                         ACTION = KNN_PCApreds$.pred_1)
vroom_write(x = KNearest_preds, file ="./AmazonKNNPCAPreds.csv", delim = ",")

