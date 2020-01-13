library(dplyr)
library(ggplot2)
library(keras)

hr <- read.csv("data/train_LZdllcl.csv", stringsAsFactors = FALSE)
hr[hr$education == "", "education"] <- "no_education/information"
hr[hr$education == "Bachelor's", "education"] <- "bachelor"
hr[hr$education == "Below Secondary", "education"] <- "below_secondary"
hr[hr$education == "Master's & above", "education"] <- "master_and_above"
hr[is.na(hr$previous_year_rating), "previous_year_rating"] <- 0


x <- hr %>% group_by(department) %>% summarise(z = mean(is_promoted), smart_index = mean(avg_training_score))
hr <- merge(hr, x[, c("department", "smart_index")], all.x = TRUE)
hr$smart_index <- hr$avg_training_score/hr$smart_index

department_input <- layer_input(shape = 1, name = "department")
region_input <- layer_input(shape = 1, name = "region")
education_input <- layer_input(shape = 1, name = "education")
channel_input <- layer_input(shape = 1, name = "channel")
training_input <- layer_input(shape = 1, name = "no_of_training")
prev_year_rating_input <- layer_input(shape = 1, name = "prev_year_rating")
other_vars_input <- layer_input(shape = 12, name = "other_vars")


dep_layer <- department_input %>%
  layer_embedding(input_dim = 9 + 1, 
                  output_dim = 5, 
                  input_length = 1, 
                  name = "department_embedding", 
                  trainable = TRUE) %>%
  layer_flatten()

reg_layer <- region_input %>%
  layer_embedding(input_dim = 34 + 1, 
                  output_dim = 17, 
                  input_length = 1, 
                  name = "region_embedding", 
                  trainable = TRUE) %>%
  layer_flatten()

edu_layer <- education_input %>%
  layer_embedding(input_dim = 4 + 1, 
                  output_dim = 3, 
                  input_length = 1, 
                  name = "education_embedding", 
                  trainable = TRUE) %>%
  layer_flatten()

chan_layer <- channel_input %>%
  layer_embedding(input_dim = 3 + 1, 
                  output_dim = 2, 
                  input_length = 1, 
                  name = "rec_channel_embedding", 
                  trainable = TRUE) %>%
  layer_flatten()

train_layer <- training_input %>%
  layer_embedding(input_dim = 6 + 1, 
                  output_dim = 3, 
                  input_length = 1, 
                  name = "no_of_trainings_embedding", 
                  trainable = TRUE) %>%
  layer_flatten()

rating_layer <- prev_year_rating_input %>%
  layer_embedding(input_dim = 6 + 1, 
                  output_dim = 4, 
                  input_length = 1, 
                  name = "prev_year_rating_embedding", 
                  trainable = TRUE) %>%
  layer_flatten()

# Concatenating model and adding more layers
combined_model <- layer_concatenate(c(dep_layer, 
                                      reg_layer, 
                                      edu_layer, 
                                      chan_layer, 
                                      train_layer, 
                                      rating_layer, 
                                      other_vars_input), name = "concatination_layer") %>%
  layer_dense(256, activation = "relu") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(5, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs = c(department_input,
                                region_input,
                                education_input,
                                channel_input,
                                training_input,
                                prev_year_rating_input,
                                other_vars_input),
                     outputs = combined_model)

inputs <- list(as.matrix(as.integer(as.factor(hr$department))),
               as.matrix(as.integer(as.factor(hr$region))),
               as.matrix(as.integer(as.factor(hr$education))),
               as.matrix(as.integer(as.factor(hr$recruitment_channel))),
               as.matrix(as.integer(as.factor(hr$no_of_trainings))),
               as.matrix(as.integer(as.factor(hr$previous_year_rating))),
               as.matrix(hr[, c("smart_index",
                                "gender",
                                "age",
                                "length_of_service", 
                                "KPIs_met..80.", 
                                "awards_won.", 
                                "avg_training_score", 
                                "is_new_employee", 
                                "age_index", 
                                "age_index1", 
                                "age_index2", 
                                "age_started_work")]))

opt <- optimizer_adadelta()

model %>%
  compile(
    optimizer = opt,
    loss = "binary_crossentropy",
    metrics = list("accuracy")
    )

model %>%
  fit(
    x = inputs,
    y = as.matrix(hr$is_promoted),
    epochs = 350,
    batch_size = 128
  )

