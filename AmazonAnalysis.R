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
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" variable
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

head(final_wf)


























