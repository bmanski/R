# Kaggle - Mechanisms of Action (MoA) Prediction
# Benjamin Manski
# 12/08/2020


# Installing packages
library(dplyr)
library(data.table)
library(tidyverse)
library(devtools)
library(keras)
install_keras()
devtools::install_github("rstudio/keras", force = T)
library(tensorflow)
install_tensorflow()

# Importing Datasets
train_features = read.csv("~/Predictive STAT 488/lish-moa/train_features.csv")
train_targets = read.csv("~/Predictive STAT 488/lish-moa/train_targets_scored.csv")
test_features = read.csv("~/Predictive STAT 488/lish-moa/test_features.csv")
sample_sub = read.csv("~/Predictive STAT 488/lish-moa/sample_submission.csv")


# Data Cleaning/Manipulation
train = as.data.frame(train_features)
test = as.data.frame(test_features)

train$cp_type = ifelse(train$cp_type == "trt_cp",1,0)
train$cp_dose = ifelse(train$cp_dose == "D1",1,0)
test$cp_type = ifelse(test$cp_type == "trt_cp",1,0)
test$cp_dose = ifelse(test$cp_dose == "D1",1,0)

train$cp_time = train$cp_time/24 - 1
test$cp_time = test$cp_time/24 - 1


train$sig_id = NULL
train_targets$sig_id = NULL
sig_id = test$sig_id
test$sig_id = NULL

x_train = data.matrix(train)
y_train = data.matrix(train_targets)
test = data.matrix(test)


# Neural Network
model = keras_model_sequential()

model %>%
  layer_dense(units = 512, activation = 'relu', input_shape = ncol(x_train))%>%
  layer_dropout(rate = .2)%>%
  layer_batch_normalization()%>%
  layer_dense(units = 256, activation = 'relu')%>%
  layer_dropout(rate = .2)%>%
  layer_batch_normalization()%>%
  layer_dense(units = 128, activation = 'relu')%>%
  layer_dropout(rate = .2)%>%
  layer_batch_normalization()%>%
  layer_dense(units = ncol(y_train), activation = 'sigmoid')


model %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'adam',
  metrics = 'accuracy')


history = model %>% fit(
  x_train,
  y_train,
  epochs = 300,
  batch_size = 2300,
  validation_split = 0.3)


pred = model$predict(test)
pred_df = as.data.frame(pred)
submit = data.frame(sig_id, pred_df)
colnames(submit) = colnames(sample_sub)

write_csv(submit, file = "./submission.csv")

# Score: 0.02270
