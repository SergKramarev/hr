# Script for inputs creation
create_inputs <- function(dataset = NULL){
  return(
    list(
      as.matrix(as.integer(as.factor(dataset$department))),
      as.matrix(as.integer(as.factor(dataset$region))),
      as.matrix(as.integer(as.factor(dataset$education))),
      as.matrix(as.integer(as.factor(dataset$recruitment_channel))),
      as.matrix(as.integer(as.factor(dataset$no_of_trainings))),
      as.matrix(as.integer(as.factor(dataset$previous_year_rating))),
      as.matrix(as.integer(as.factor(dataset$age_group))),
      as.matrix(as.integer(as.factor(dataset$length_group))),
      as.matrix(dataset[, c("smart_index",
                            "gender",
                            "KPIs_met..80.", 
                            "awards_won.", 
                            "avg_training_score", 
                            "is_new_employee",
                            "previous_year_index",
                            #"length_index1",
                            "length_index2",
                            #"age_index1",
                            "age_index2",
                            "region_index"
                          )]))
  )
}