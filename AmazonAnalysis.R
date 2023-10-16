library(vroom)
library(tidymodels)
library(embed)
library(DataExplorer)
library(GGally)


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















