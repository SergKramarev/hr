source("prepare_dataset.R")
source("create_inputs.R")
library(keras)
library(tensorflow)

# Setting memory growth == TRUE allows to train nets in background 
gpu <- tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu, enable = TRUE)

# Reading dataset
hr <- read.csv("data/train_LZdllcl.csv", stringsAsFactors = FALSE)

train_ind <- sample(1:nrow(hr), nrow(hr)*0.7, replace = FALSE)

train_set <- hr[train_ind, ]
test_set <- hr[-train_ind, ]

# Dataset preparation
train_set <- prepare_dataset(train_set, testing_set = FALSE)
test_set <- prepare_dataset(test_set, testing_set = TRUE)

# Creating inputs for neural net
inputs_train <- create_inputs(train_set)
inputs_test <- create_inputs(test_set)

# Creating inputs to neural net  
department_input <- layer_input(shape = 1, name = "department")
region_input <- layer_input(shape = 1, name = "region")
education_input <- layer_input(shape = 1, name = "education")
recruitment_input <- layer_input(shape = 1, name = "recruitment")
training_input <- layer_input(shape = 1, name = "no_of_training")
previous_rating_input <- layer_input(shape = 1, name = "previous_rating")
age_input <- layer_input(shape = 1, name = "age")
length_input <- layer_input(shape = 1, name = "length")
other_vars_input <- layer_input(shape = 11, name = "other_vars")


# Creating layers
dep_layer <- department_input %>%
  layer_embedding(input_dim = 9 + 1, output_dim = 5, input_length = 1,
                  trainable = TRUE, batch_size = 2) %>%
  layer_flatten()

reg_layer <- region_input %>%
  layer_embedding(input_dim = 34 + 1, output_dim = 18, input_length = 1, 
                  trainable = TRUE, batch_size = 2) %>%
  layer_flatten()

edu_layer <- education_input %>%
  layer_embedding(input_dim = 4 + 1, output_dim = 2, input_length = 1, 
                  trainable = TRUE, batch_size = 2) %>%
  layer_flatten()

recruit_layer <- recruitment_input %>%
  layer_embedding(input_dim = 3 + 1, output_dim = 2, input_length = 1, 
                  trainable = TRUE, batch_size = 2) %>%
  layer_flatten()

train_layer <- training_input %>%
  layer_embedding(input_dim = 6 + 1, output_dim = 3, input_length = 1, 
                  trainable = TRUE, batch_size = 2) %>%
  layer_flatten()

previous_rating_layer <- previous_rating_input %>%
  layer_embedding(input_dim = 6 + 1, output_dim = 4, input_length = 1, 
                  trainable = TRUE, batch_size = 2) %>%
  layer_flatten()

age_layer <- age_input %>%
  layer_embedding(input_dim = 30 + 1, output_dim = 15, input_length = 1, 
                  trainable = TRUE, batch_size = 2) %>%
  layer_flatten()

length_layer <- length_input %>%
  layer_embedding(input_dim = 16 + 1, output_dim = 8, input_length = 1, 
                  trainable = TRUE, batch_size = 2) %>%
  layer_flatten()

# Concatenating layers
combined_model <- layer_concatenate(c(dep_layer, 
                                      reg_layer, 
                                      edu_layer, 
                                      recruit_layer, 
                                      train_layer,
                                      previous_rating_layer,
                                      age_layer,
                                      length_layer,
                                      other_vars_input), name = "concatination_layer") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

# Creating model
model <- keras_model(inputs = c(department_input,
                                region_input,
                                education_input,
                                recruitment_input,
                                training_input,
                                previous_rating_input,
                                age_input,
                                length_input,
                                other_vars_input),
                     outputs = combined_model)
  
opt <- optimizer_adadelta()

model %>%
    compile(optimizer = opt,
            loss = "binary_crossentropy",
            metrics = "acc")

model %>%
    fit(x = inputs_train,
        y = as.matrix(train_set$is_promoted),
        epochs = 50,
        batch_size = 128,
        callbacks = callback_early_stopping(monitor = "val_loss", min_delta = 0.0001, patience = 10, restore_best_weights = TRUE),
        validation_data = list(inputs_test, as.matrix(test_set$is_promoted)))