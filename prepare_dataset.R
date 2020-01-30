prepare_dataset <- function(dataset = hr, testing_set = FALSE){
  require(dplyr)
  
  quant_age <- c(0, 20, 21, 22, 23, 25, 28, 30, 32, 35, 38, 40, 42, 46, 50, 55, 80)
  quant_length <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 21, 25, 50)
  
  dataset <- dataset %>%
    mutate(
      education = ifelse(education == "", "no_eduction/information", education),
      education = tolower(education),
      education = gsub(" ", "_", education),
      gender = as.integer(as.factor(gender)),
      is_new_employee = ifelse(is.na(previous_year_rating), 1, 0),
      age_started_work = age - length_of_service,
      no_of_trainings = ifelse(no_of_trainings %in% c(5, 6, 7, 8, 9, 10), "4+", no_of_trainings),
      age_group = cut(age, quant_age),
      length_group = cut(length_of_service, quant_length),
      )
  
  dataset <- dataset %>%
    group_by(department) %>%
    mutate(previous_year_rating = ifelse(is.na(previous_year_rating), 
                                         round(mean(previous_year_rating, na.rm = TRUE), digits = 3), 
                                         previous_year_rating)) %>%
    ungroup()

if (!testing_set){
    
    departments_indexes <<- dataset %>%
      group_by(department) %>%
      summarise(
                smart_index = mean(avg_training_score),
                award_index = 1/mean(awards_won.)^2/1000,
                KPI_index = 1/mean(KPIs_met..80.)^2/5,
                avg_previous_year_rating_by_dept = mean(previous_year_rating, na.rm = TRUE),
                number_of_observation_in_dept = n(),
                avg_is_promoted_by_dept = mean(is_promoted, na.rm = TRUE)
                )
      
    prev_year_indexes <<- dataset %>%
      group_by(department, previous_year_rating) %>%
      summarise(
                number_of_observations_by_rating_by_dept = n()
                )
   region_indexes <<- dataset %>%
     group_by(department, region) %>%
     summarise(avg_promoted_by_dept_by_region = mean(is_promoted, na.rm = TRUE),
               number_of_observations_in_region_group = n())
    
    length_group_indexes <<- dataset %>%
      group_by(department, length_group) %>%
      summarise(avg_promoted_by_dept_by_length_group = mean(is_promoted, na.rm = TRUE),
                number_of_observations_in_length_group = n()
                )
    
    age_group_indexes <<- dataset %>%
      group_by(department, age_group) %>%
      summarise(avg_is_promoted_by_dpt_by_age_group = mean(is_promoted),
                number_of_observations_in_age_group = n()
                )
  }
  
  
  dataset <- merge(dataset, departments_indexes)
  dataset <- merge(dataset, prev_year_indexes, all.x = TRUE)
  dataset <- merge(dataset, length_group_indexes, all.x = TRUE)
  dataset <- merge(dataset, age_group_indexes, all.x = TRUE)
  dataset <- merge(dataset, region_indexes, all.x = TRUE)
  
  dataset <- dataset %>%
    mutate(
      smart_index = avg_training_score/smart_index,
      awards_won. = awards_won.*award_index,
      KPIs_met..80. = KPIs_met..80.*KPI_index,
      previous_year_index = (previous_year_rating - avg_previous_year_rating_by_dept)/sqrt(number_of_observations_by_rating_by_dept/number_of_observation_in_dept),
      age_index = (avg_is_promoted_by_dpt_by_age_group - avg_is_promoted_by_dept)*10,
      length_index = (avg_promoted_by_dept_by_length_group - avg_is_promoted_by_dept)*10
      )
  

  dataset[is.na(dataset$previous_year_index), "previous_year_index"] <- 0
  dataset[is.na(dataset$age_index), "age_index"] <- 0
  dataset[is.na(dataset$length_index), "length_index"] <- 0
  dataset[is.na(dataset$region_index), "region_index"] <- 0
  
 
  dataset$region_index <- scale(dataset$region_index, center = FALSE)
 
  
  
  if (!testing_set){
      dataset$avg_training_score <- scale(dataset$avg_training_score)
  } else {
     dataset$avg_training_score <- (dataset$avg_training_score - att$avg_training_score$`scaled:center`)/att$avg_training_score$`scaled:scale`
  }
  
  if (!testing_set){
    att <<- list(
      avg_training_score = attributes(dataset$avg_training_score)
      )
  }
  
  return(dataset)
}
