install.packages("embed")
library(vroom)
library(tidymodels)
library(embed)
library(DataExplorer)
library(GGally)


#Read in the data
amazon_train <- vroom("./train.csv")
amazon_test <- vroom("./test.csv")

glimpse(amazon_train)
plot_intro(amazon_train)
plot_correlation(amazon_train)

A_Recipe <- recipe(ACTION~., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(),fn= factor) %>% # 
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" variable
  step_dummy(all_nominal_predictors())

prep <- prep(A_Recipe)
baked <- bake(prep, new_data = amazon_train)

glimpse(baked)
