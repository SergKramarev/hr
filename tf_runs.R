source("prepare_dataset.R")
source("create_inputs.R")
library(keras)
library(tensorflow)
library(tfruns)

# Setting memory growth == TRUE allows to train nets in background 
gpu <- tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu, enable = TRUE)

# Reading dataset
hr <- read.csv("data/train_LZdllcl.csv", stringsAsFactors = FALSE)
set.seed(124)
train_ind <- sample(1:nrow(hr), nrow(hr)*0.7, replace = FALSE)

train_set <- hr[train_ind, ]
test_set <- hr[-train_ind, ]

# Dataset preparation, new variables creation
train_set <- prepare_dataset(train_set, testing_set = FALSE)
test_set <- prepare_dataset(test_set, testing_set = TRUE)

# Creating inputs for neural net
inputs_train <- create_inputs(train_set)
inputs_test <- create_inputs(test_set)

# Creating inputs to neural net  
department_input <- layer_input(shape = 1, name = "department")
region_input <- layer_input(shape = 1, name = "region")
education_input <- layer_input(shape = 1, name = "education")
training_input <- layer_input(shape = 1, name = "no_of_training")
prev_year_rating_input <- layer_input(shape = 1, name = "prev_year_rating")
length_input <- layer_input(shape = 1, name = "length_of_service")
age_input <- layer_input(shape = 1, name = "age")
age_started_input <- layer_input(shape = 1, name = "age_started")
other_vars_input <- layer_input(shape = 10, name = "other_vars")
  
# Creating layers
dep_layer <- department_input %>%
    layer_embedding(input_dim = 9 + 1,
                    output_dim = 5,
                    input_length = 1,
                    name = "department_embedding",
                    trainable = TRUE, batch_size = 2) %>%
    layer_flatten()
  
reg_layer <- region_input %>%
    layer_embedding(input_dim = 34 + 1, 
                    output_dim = 17, 
                    input_length = 1, 
                    name = "region_embedding", 
                    trainable = TRUE, batch_size = 2) %>%
    layer_flatten()
  
edu_layer <- education_input %>%
    layer_embedding(input_dim = 4 + 1, 
                    output_dim = 3, 
                    input_length = 1, 
                    name = "education_embedding", 
                    trainable = TRUE, batch_size = 2) %>%
    layer_flatten()
  
 
train_layer <- training_input %>%
    layer_embedding(input_dim = 5 + 1, 
                    output_dim = 3, 
                    input_length = 1, 
                    name = "no_of_trainings_embedding", 
                    trainable = TRUE, batch_size = 2) %>%
    layer_flatten()
  
rating_layer <- prev_year_rating_input %>%
    layer_embedding(input_dim = 46 + 1, 
                    output_dim = 23, 
                    input_length = 1, 
                    name = "prev_year_rating_embedding", 
                    trainable = TRUE, batch_size = 2) %>%
    layer_flatten()
  
length_layer <- length_input %>%
    layer_embedding(input_dim = 35 + 1, 
                    output_dim = 18, 
                    input_length = 1, 
                    name = "length_embedding", 
                    trainable = TRUE, batch_size = 2) %>%
    layer_flatten()
  
age_layer <- age_input %>%
    layer_embedding(input_dim = 18 + 1, 
                    output_dim = 10, 
                    input_length = 1, 
                    name = "age_embedding", 
                    trainable = TRUE, batch_size = 2) %>%
    layer_flatten()
  
age_started_layer <- age_started_input %>%
    layer_embedding(input_dim = 42 + 1, 
                    output_dim = 22, 
                    input_length = 1, 
                    name = "age_started_embedding", 
                    trainable = TRUE, batch_size = 2) %>%
    layer_flatten()
  
# Concatenating layers and adding more 
combined_model <- layer_concatenate(c(dep_layer, 
                                     #  reg_layer, 
                                        edu_layer, 
                                       # train_layer, 
                                       # age_started_layer,
                                        other_vars_input), name = "concatination_layer") %>%
    layer_dense(128, activation = "relu") %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(1, activation = "sigmoid")
  
model <- keras_model(inputs = c(department_input,
                               #  region_input,
                                  education_input,
                                #  training_input,
                                 # age_started_input,
                                  other_vars_input),
                    outputs = combined_model)
  
opt <- optimizer_adadelta()

model %>%
    compile(
      optimizer = opt,
      loss = "binary_crossentropy",
      metrics = c("accuracy"))
  

model %>%
    fit(
      x = inputs_train,
      y = as.matrix(train_set$is_promoted),
      epochs = 20,
      batch_size = 128,
      callbacks = callback_early_stopping(monitor = "val_loss", min_delta = 0.0001, patience = 10, restore_best_weights = TRUE),
      validation_data = list(inputs_test, as.matrix(test_set$is_promoted))
    )

  # train_set$nn_prediction <- predict(model, inputs_train)
  # test_set$nn_prediction <- predict(model, inputs_test)

  
  
  
 