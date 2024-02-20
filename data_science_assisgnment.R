
#cleaning the environment.
rm( list = ls())
#loading the package
library(tidyverse)
library(dplyr)
library(ggplot2)
library(fpp2)
library(glmnet)
library(tidyr)
library(lmtest)
library(boot)
library(forecast)
library(readr)
library(ggfortify)
library(tseries)
library(urca)
library(readxl)
library(lubridate)
library(tsbox)
library(RColorBrewer)
library(wesanderson)
library(writexl)
library(gridExtra)
library(vars)
library(leaps)
library(broom)
library(fastDummies)
####################
#The R package her just copy and past from my undergraduate assignment 
#As I know or used most command come form 
###################

#0.1 finished the first round of linear regression model
#0.2 add the age specifc part for teh linear regression


##################



#setting the saving addreess
setwd("/Users/tie/Documents/GitHub/The-data-analysis-job--")

#####################
#Step one: Clean the data
#####################
  Titanic_train_raw <- read_csv("Titanic data/train.csv", 
                                col_types = cols( Name = col_skip(), Ticket = col_skip(), 
                                                 Cabin = col_skip()))
  
#At here I skiped their name which does not impact the model training, 
#their ticket number with is not important and their cabin which has limit number


print(colSums(is.na(Titanic_train_raw )))

################ The age gap. 
#I notice there are some age part are empty so i decided to use mean of age to fill the gap. 
#but before that I just need to check the distribution of age between the survived and died


#check the age of average by the survive or not 
Titanic_train_raw %>% group_by(Survived) %>% summarise(mean(Age, na.rm = TRUE)) 

#calcuate the average of all the age 
The_average_age <- mean(Titanic_train_raw$Age, na.rm = TRUE)
#print(The_average_age)

#they are close enough, now fill the age gap. 
#fill the "age gap"
Titanic_train_raw$Age[is.na(Titanic_train_raw$Age)] <- The_average_age
Titanic_train_without_age_gap <- Titanic_train_raw

#head()
#head(Titanic_train_without_age_gap)

#check again
print(colSums(is.na(Titanic_train_without_age_gap)))


#delete the 2 line in the embarked 
Titanic_train <- Titanic_train_without_age_gap[!is.na(Titanic_train_without_age_gap$Embarked), ]
Titanic_train_cleaned <- na.omit(Titanic_train_without_age_gap)

#delete all the data that including at least one NA value

#Change all the gender to the dummy variable 
#male = 1 female equal to zero 

TTD<- dummy_cols(Titanic_train_cleaned, select_columns = "Sex", remove_selected_columns = TRUE)


################

#head(Titanic_train)
#its look good

######################
#Step two: Sharking method
######################

#using the ridge regession to search all the possible linear combination 

#crate a new matrix 
x <- TTD %>%
  dplyr::select(
    Pclass, Age, SibSp, Parch, Fare, Embarked, Sex_female
  ) %>%
  data.matrix()

#take out the survive
live <- TTD$Survived

#put two things together as data frame
data2 <- data.frame(x, live)

#using regression to search all the possible output
regfit_all <- regsubsets(live ~ ., data = data2, nvmax = 10)
reg_summary <- summary(regfit_all)

# Reset the plotting layout to default
par(mfrow=c(1,1))
plot(regfit_all, scale = "bic")


########## using the sharking method to determind the optimual output for the number of variable
# Set up plotting area to display three plots in one row
par(mfrow=c(1,3))

# Plot RSS
plot(reg_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")

# Plot BIC with highlighted minimum point
plot(reg_summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
m.bic <- which.min(reg_summary$bic) # Find the index of minimum BIC
points(m.bic, reg_summary$bic[m.bic], col = "red", cex = 2, pch = 20) # Highlight the min point

# Plot Cp with highlighted minimum point
plot(reg_summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
m.cp <- which.min(reg_summary$cp) # Find the index of minimum Cp
points(m.cp, reg_summary$cp[m.cp], col = "red", cex = 2, pch = 20) # Highlight the min point


 

############
#According to the ridge regression, I should choose two group as the following variable
# the first group
  #Pclass, Age, sibsp, Sex_female
# the second group
  #Pclass, Age, sibsp, Embarked, Sex_female 


##########The lasso regresson time
# Fit LASSO model with cross-validation
lasso.cv <- cv.glmnet(x, live, alpha = 1, nfolds = 5)

# Extract LASSO coefficients at the optimal lambda found by cross-validation
coef_lasso <- as.vector(coef(lasso.cv, s = "lambda.min")[-1])  # Excludes intercept

# Fit OLS model using lm()
ols_mod <- lm(live ~ ., data = data2)

# Extract coefficients from the OLS model (excluding intercept for direct comparison)
coef_ols <- coef(ols_mod)[-1]  # Exclude intercept

# Best subset selection using regsubsets() from the leaps package
best_subset_selection <- regsubsets(live ~ ., data = data2, nvmax = 10)
best_subset_coef <- coef(best_subset_selection, id = which.min(summary(best_subset_selection)$bic))

# Create a data frame for comparison
# Since variable names in LASSO and OLS are the same, use those for the row names
variable_names <- names(coef_ols)

# For best subset selection, extract coefficients including intercept (adjust accordingly if needed)
coef_bss <- as.vector(best_subset_coef)  # Adjust this line based on how coef_bss is structured

# Initialize a comparison table
comparison_table <- data.frame(
  Variable = variable_names,
  OLS = coef_ols,
  LASSO = coef_lasso
)

# Display the comparison table
comparison_table_sorted <- comparison_table[order(-comparison_table$LASSO), ]

# Display the sorted comparison table
print(comparison_table_sorted)

#The lasso regression does not sharking enough meanslasso regression found
#all the variable are actually have some degree of importantence. 


#according the lasso regression we should use the 
  #Sex and Pclass

############ OLS model time
# the first group
#Pclass, Age, sibsp, Sex_female
# the second group
#Pclass, Age, sibsp, Embarked, Sex_female 


########### 
#using the 2/3 of training data to traning and using 1/3 of the training data to test
#which group has the lowest BIC which group win!


#TTD

# Split the data
training_indices <- sample(1:nrow(TTD), size = 2/3 * nrow(TTD))
training_data <- TTD[training_indices, ]
testing_data <- TTD[-training_indices, ]

# Fit OLS models for each group on the training data
model_group1 <- lm(Survived ~ Pclass + Sex_female + Age + SibSp, data = TTD)
model_group2 <- lm(Survived ~ Pclass + Sex_female + Age + SibSp + Embarked, data =TTD)

# Making predictions on the test set for each model group
predictions_group1 <- predict(model_group1, newdata = testing_data, type = "response")
predictions_group2 <- predict(model_group2, newdata = testing_data, type = "response")

# Actual values
actuals <- testing_data$Survived

# Calculate MSE, RMSE, and MAE for each group
calculate_metrics <- function(actuals, predictions) {
  mse <- mean((actuals - predictions)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(actuals - predictions))
  return(c(MSE = mse, RMSE = rmse, MAE = mae))
}

metrics_group1 <- calculate_metrics(actuals, predictions_group1)
metrics_group2 <- calculate_metrics(actuals, predictions_group2)


# Compile and display the results
results <- rbind(Group1 = metrics_group1, Group2 = metrics_group2)
print(results)

#according the cut sample testing the group 1 with!
#The model we should use
#The winning linear regression model!

Winning_model <- lm(Survived ~ Pclass + Sex_female + Age + SibSp + Embarked, data =TTD)


#The time for the test set!
test_data_set <- read_csv("Titanic data/test.csv", 
                 col_types = cols(Name = col_skip(), Parch = col_skip(), 
                                  Ticket = col_skip(), Fare = col_skip(), 
                                  Cabin = col_skip()))
head(test_data_set)

#check if there any NA here
  #the age have 86 empty 
  #using the average to replace it. 


# Fill in missing Age values with the mean Age, ensuring NA values are handled
test_data_set_without_age_gap <- test_data_set %>%
  mutate(Age = ifelse(is.na(Age), mean(Age, na.rm = TRUE), Age))

#Transfer the sex to dummy variable 
test_data_ultra<- dummy_cols(test_data_set_without_age_gap, select_columns = "Sex", remove_selected_columns = TRUE)



# Predict survival probabilities
predicted_probabilities <- predict(Winning_model, newdata = test_data_ultra, type = "response")

# Convert probabilities to binary predictions
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)

# Add Predicted_Survived to the dataset
test_data_set_without_age_gap$Predicted_Survived <- predicted_classes

# Select only PassengerId and Predicted_Survived

The_final_test <- cbind(test_data_set_without_age_gap$PassengerId, test_data_set_without_age_gap$Predicted_Survived )



# Save to CSV
write.csv(The_final_test, "The_final_test.csv", row.names = FALSE)




