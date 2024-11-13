library(dplyr)
library(caret)
library(randomForest)
library(pROC)

accepted_data <- read.csv("C:/Users/lenovo/Desktop/data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv")


accepted_data$loan_status <- ifelse(accepted_data$loan_status == "Charged Off", 1, 0)


loan_status_distribution <- table(accepted_data$loan_status)
print(loan_status_distribution)


if (length(unique(accepted_data$loan_status)) < 2) {
  stop("Error: loan_status must have at least two classes for training.")
}

accepted_data <- accepted_data %>% 
  select(loan_amnt, term, int_rate, grade, emp_length, annual_inc, loan_status) %>%
  na.omit()

#Encode
accepted_data$term <- as.numeric(factor(accepted_data$term))
accepted_data$grade <- as.numeric(factor(accepted_data$grade))
accepted_data$emp_length <- as.numeric(factor(accepted_data$emp_length))

accepted_data <- accepted_data %>% mutate(across(c(loan_amnt, int_rate, annual_inc), scale))


set.seed(123)

trainIndex <- createDataPartition(accepted_data$loan_status, p = 0.8, list = FALSE)
train_data <- accepted_data[trainIndex, ]
test_data <- accepted_data[-trainIndex, ]


set.seed(123)
rf_model <- randomForest(loan_status ~ ., data = train_data, ntree = 100, mtry = 3)
print(rf_model)

test_predictions <- predict(rf_model, test_data)
conf_matrix <- confusionMatrix(as.factor(test_predictions), as.factor(test_data$loan_status))
print(conf_matrix)


roc_curve <- roc(test_data$loan_status, as.numeric(test_predictions))
auc_value <- auc(roc_curve)
print(auc_value)
plot(roc_curve, col = "blue", main = "ROC Curve for Loan Default Prediction")

#plot
importance <- importance(rf_model)
print(importance)
varImpPlot(rf_model, main = "Feature Importance for Loan Default Prediction")