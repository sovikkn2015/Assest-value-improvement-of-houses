library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
#library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization
library(tidyverse)

options("scipen"=100, "digits"=4)

# ############################# -------------------------------------------

data_df = read.csv("Redlands Simplified.csv", header = TRUE, sep = ",")
colnames_df <- colnames(data_df)

targetAPN <- data_df[data_df$apn == "175111130000", ]
data_df2<- data_df[-c(data_df$apn == "175111130000"), ]
data_df<- data_df2

selectvars <- c("number_of_buildings","situs_zip_code","assessed_land_value","land_square_footage",
                "year_built","bedrooms_all_buildings","total_rooms_all_buildings","total_bath_rooms_calculated_all_buildings",
                "total_baths_primary_building","full_baths_all_buildings","half_baths_all_buildings","pool_indicator",
                "stories_number","universal_building_square_feet","building_square_feet","living_square_feet_all_buildings",
                "ground_floor_square_feet","basement_square_feet","garage_parking_square_feet","avm_value_amount")
select_df <- data_df[selectvars]

select_df[is.na(select_df)] <- 0
#sapply(select_df, class)

select_df$pool_indicator <- as.character(select_df$pool_indicator)
select_df$pool_indicator[select_df$pool_indicator == "N"] <- 0
select_df$pool_indicator[select_df$pool_indicator == "Y"] <- 1
select_df$pool_indicator <- as.integer(select_df$pool_indicator)
select_df[is.na(select_df)] <- 0


# ############################# -------------------------------------------

# Seperate the data based on zipcode into 2 clusters

select_df_set1 <- select_df[which(select_df$situs_zip_code > 923720000 & select_df$situs_zip_code < 923740000), ]
select_df_set2 <- select_df[which(select_df$situs_zip_code >= 923740000), ]

# ############################# -------------------------------------------


set.seed(123)
ames_split_set1 <- initial_split(select_df_set1, prop = .9)
ames_train_set1 <- training(ames_split_set1)
ames_test_set1  <- testing(ames_split_set1)

set.seed(123)
ames_split_set2 <- initial_split(select_df_set2, prop = .9)
ames_train_set2 <- training(ames_split_set2)
ames_test_set2  <- testing(ames_split_set2)


# ############################# -------------------------------------------
# 
# # for reproducibility
# set.seed(123)
# 
# # train GBM model
# gbm.fit <- gbm(
#   formula = avm_value_amount ~ .,
#   distribution = "gaussian",
#   data = ames_train,
#   n.trees = 10000,
#   interaction.depth = 1,
#   shrinkage = 0.001,
#   cv.folds = 5,
#   n.cores = NULL, # will use all cores by default
#   verbose = FALSE
# )  
# 
# # print results
# print(gbm.fit)
# 
# # get MSE and compute RMSE
# sqrt(min(gbm.fit$cv.error))
# 
# # plot loss function as a result of n trees added to the ensemble
# gbm.perf(gbm.fit, method = "cv")
# 
# 
# # ############################# -------------------------------------------
# 
# ### TUNING
# 
# # for reproducibility
# set.seed(123)
# 
# # train GBM model
# gbm.fit2 <- gbm(
#   formula = avm_value_amount ~ .,
#   distribution = "gaussian",
#   data = ames_train,
#   n.trees = 5000,
#   interaction.depth = 3,
#   shrinkage = 0.1,
#   cv.folds = 5,
#   n.cores = NULL, # will use all cores by default
#   verbose = FALSE
# )  
# 
# # find index for n trees with minimum CV error
# min_MSE <- which.min(gbm.fit2$cv.error)
# 
# # get MSE and compute RMSE
# sqrt(gbm.fit2$cv.error[min_MSE])
# 
# # plot loss function as a result of n trees added to the ensemble
# gbm.perf(gbm.fit2, method = "cv")


# ############################# -------------------------------------------

## SET 1

# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.075, .1, .125),
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 8, 10),
  bag.fraction = c(.8, 1, 1.2), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)

# randomize data
random_index_set1 <- sample(1:nrow(ames_train_set1), nrow(ames_train_set1))
random_ames_train_set1 <- ames_train_set1[random_index_set1, ]

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = avm_value_amount ~ .,
    distribution = "gaussian",
    data = random_ames_train_set1,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid <- hyper_grid %>% 
  dplyr::arrange(min_RMSE) 

write.csv(hyper_grid,"gbm_hypergrid_3_set1.csv", row.names = FALSE)


# ############################# -------------------------------------------

## TUNED MODEL

# for reproducibility
set.seed(123)

# train GBM model
gbm.fit.final <- gbm(
  formula = avm_value_amount ~ .,
  distribution = "gaussian",
  data = ames_train_set1,
  n.trees = hyper_grid$optimal_trees[1],
  interaction.depth = hyper_grid$interaction.depth[1],
  shrinkage = hyper_grid$shrinkage[1],
  n.minobsinnode = hyper_grid$n.minobsinnode[1],
  bag.fraction = hyper_grid$bag.fraction[1], 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  


# ############################# -------------------------------------------

# VISUALIZING

#par(mar = c(5, 8, 1, 1))
gbm_fit_summary <- summary(
  gbm.fit.final, 
  cBars = 19,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

gbm_fit_summary

write.csv(gbm_fit_summary,"gbm_fit_summary3_set1.csv", row.names = FALSE)

#write.csv(gbm.fit.final,"gbm.fit.final3_set1.csv", row.names = FALSE)


# ############################# -------------------------------------------

# predict values for test data
predgbm <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, ames_test_set1)

# results
rmsegbm <- caret::RMSE(predgbm, ames_test_set1$avm_value_amount)

write.csv(rmsegbm,"rmsegbm3_set1.csv", row.names = FALSE)

# # ############################# -------------------------------------------

saveRDS(gbm.fit.final, "avm_gbm_set1.rds")
#my_model <- readRDS("avm_gbm.rds")


# # ############################# -------------------------------------------
# 
# ## CROSS VALIDATION
# 
# # set seed for reproducibility
# set.seed(256)
# # create 5 folds to be used in cross validation
# myFolds <- createFolds(ames_train, k = 5)
# # create a custom trainControl object to use our folds; index = myFolds
# myControl = trainControl(verboseIter = FALSE, index = myFolds)
# 
# # ############################# -------------------------------------------
# 
# 
# ## KNN
# 
# set.seed(579)
# # Train glmnet with custom trainControl and tuning: model
# knn1 <- train(
#   # formula
#   avm_value_amount ~ ., 
#   # data
#   ames_train,
#   # knn regression
#   method = "kknn",
#   # trainControl
#   trControl = myControl
# )
# 
# print(knn1)
# 
# # ############################# -------------------------------------------
# 
# ## RANDOM FOREST
# 
# 
# library(randomForest)
# ## randomForest 4.6-12
# ## Type rfNews() to see new features/changes/bug fixes.
# set.seed(1)
# bag.avm=randomForest(avm_value_amount~.,data=ames_train,mtry=19,importance=TRUE)
# bag.avm
# 
# #yhat.bag = predict(bag.avm,newdata=select_df[-train,])
# 
# # predict values for test data
# predrf <- predict(bag.avm,newdata=ames_test)
# 
# # results
# rmserf <-caret::RMSE(predrf, ames_test$avm_value_amount)
# plot(pred, ames_test$avm_value_amount)
# abline(0,1)
# 
# importance(bag.avm)
# varimp_df<-varImpPlot(bag.avm)
# 
# write.csv(varimp_df,"varimp_df2.csv", row.names = FALSE)
# 
# 
# write.csv(rmserf ,"rmserf2.csv", row.names = FALSE)
# 
# 
# # ############################# -------------------------------------------
# 
# ## RIDGE
# 
# set.seed(1267)
# # Train glmnet with custom trainControl and tuning: model
# ridge <- train(
#   # formula
#   avm_value_amount ~ ., 
#   # data
#   ames_train,
#   # set grid search parameters for lambda
#   tuneGrid = expand.grid(alpha = 1, 
#                          lambda = (0:15) * 1000),
#   # use glmnet method for lasso, ridge, and elastic net 
#   method = "glmnet",
#   # trainControl
#   trControl = myControl
# )
# # Print model output to console
# print(ridge)
# 
# # ############################# -------------------------------------------
# 
# ## AUTOML
# 
# library(automl)
# 
# amlmodel = automl_train_manual(Xref = subset(ames_train, select = -c(avm_value_amount)),
#                                Yref = subset(ames_train, select = c(avm_value_amount))$avm_value_amount
#                                %>% as.numeric(),
#                                hpar = list(learningrate = 0.01,
#                                            minibatchsize = 2^2,
#                                            numiterations = 60))
# 
# predautoml = automl_predict(model = amlmodel, X = ames_test) 
# 
# #prediction = ifelse(prediction > 2.5, 3, ifelse(prediction > 1.5, 2, 1)) %>% as.factor()
# 
# rmseautoml <-caret::RMSE(predautoml, ames_test$avm_value_amount)
# 
# write.csv(rmseautoml ,"rmseautoml2.csv", row.names = FALSE)
# 
# # ############################# -------------------------------------------
# 
