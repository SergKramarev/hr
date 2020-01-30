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
education_input <- layer_input(shape = 1, name = "education")
other_vars_input <- layer_input(shape = 10, name = "other_vars")
  
# Creating layers
dep_layer <- department_input %>%
    layer_embedding(input_dim = 9 + 1,
                    output_dim = 5,
                    input_length = 1,
                    name = "department_embedding",
                    trainable = TRUE, batch_size = 2) %>%
    layer_flatten()
  
edu_layer <- education_input %>%
    layer_embedding(input_dim = 4 + 1, 
                    output_dim = 3, 
                    input_length = 1, 
                    name = "education_embedding", 
                    trainable = TRUE, batch_size = 2) %>%
    layer_flatten()

FLAGS <- list(
  flag_numeric("regul_l1", 0),
  flag_numeric("regul_l2", 0)
)

# Concatenating layers and adding more 
combined_model <- layer_concatenate(c(dep_layer, 
                                      edu_layer, 
                                      other_vars_input), name = "concatination_layer") %>%
    layer_dense(64, activation = "relu", kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$regul_l1, l2 = FLAGS$regul_l2)) %>%
    layer_dense(32, activation = "relu", kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$regul_l1, l2 = FLAGS$regul_l2)) %>%
    layer_dense(32, activation = "relu", kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$regul_l1, l2 = FLAGS$regul_l2)) %>%
    layer_dense(1, activation = "sigmoid")
  
model <- keras_model(inputs = c(department_input,
                                education_input,
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
      epochs = 100,
      batch_size = 128,
      callbacks = callback_early_stopping(monitor = "val_loss", min_delta = 0.0001, patience = 50, restore_best_weights = TRUE),
      validation_data = list(inputs_test, as.matrix(test_set$is_promoted))
    )

  # train_set$nn_prediction <- predict(model, inputs_train)
  # test_set$nn_prediction <- predict(model, inputs_test)

  
  
  
 