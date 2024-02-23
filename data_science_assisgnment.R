
#cleaning the environment.
rm( list = ls())


#loading the package
library(tidyverse)
library(dplyr)
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
library( lubridate)
library(tsbox)
library(RColorBrewer)
library(wesanderson)
library(writexl)
library(gridExtra)
library(vars)
library(leaps)
library(broom)
library(fastDummies)
library(car)

####################
#The R package her just copy and past from my undergraduate assignment 
#As where all my command I know come from
###################

#Log
#0.1 finished the first round of linear regression model
#0.2 add the age specif part for the linear regression
#0.3 add the data examing part
#0.4 start the final process

##################

#setting the saving addreess
setwd("/Users/tie/Documents/GitHub/The-data-analysis-job--")

######################
#Step one: Clean the data
######################
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
mean_age_by_survival <- aggregate(Age ~ Survived, data = Titanic_train_raw, FUN = function(x) mean(x, na.rm = TRUE))

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




######################
#step two: check the data#####
par(mfrow = c(3, 2))

#the grpahy per class
barplot(table(TTD$Survived, TTD$Pclass), beside = TRUE, 
        main = "Survived by Pclass", xlab = "Pclass", ylab = "Count")
legend("top", legend = c("0: Died", "1: Survived"), fill = c("black", "white"))


# Age Boxplot
boxplot(Age ~ Survived, data = TTD, 
        col = c("orange", "gray"), 
        main = "Survived by Age", 
        xlab = "Survived", ylab = "Age")
legend("top", legend = c("0: Died", "1: Survived"), fill = c("orange", "gray"))


# SibSp
survival_table_sibsp <- table(TTD$Survived, TTD$SibSp)
barplot(survival_table_sibsp, beside = TRUE, col = c("black", "gray"), 
        main = "Survived by SibSp", xlab = "SibSp", ylab = "Count")
legend("topright", legend = c("0: Died", "1: Survived"), fill = c("black", "gray"))


# The parch
stripchart(Parch ~ Survived, data = TTD, vertical = TRUE, method = "jitter", 
           pch = 20, col = c("black", "gray"),
           main = " Survived by parch",
           xlab = "Survived", ylab = "Parch")
legend("top", legend = c("0: Died", "1: Survived"), fill = c("black", "gray"))

# The fare(The price of ticket)
boxplot(Fare ~ Survived, data = TTD, 
        col = c("orange", "gray"), 
        main = "Survived by Fare",
        xlab = "Survived", ylab = "Fare")
legend("top", legend = c("0: Died", "1: Survived"), fill = c("black", "gray"))

#The sex

female_table <- table(TTD$Sex_female, TTD$Survived)

barplot(female_table, beside = TRUE, col = c("black", "gray"),
        main = "Survived by Sex", xlab = "Sex", ylab = "Count",
        names.arg = c("Male", "Female"),
        legend = TRUE)
legend("topright", legend = c("Died", "Survived"), fill = c("black", "gray"))


######## Test of the heteroscedasticity.

#crate a new matrix 
x <- TTD%>%
  dplyr::select(
    Pclass, Age, SibSp, Parch, Fare, Embarked, Sex_female
  ) %>%
  data.matrix()

#take out the survive
live <- TTD$Survived

#put two things together as data frame
data2 <- data.frame(x, live)

#The BP test
data_test_1 <- lm(live ~ . , data = data2)
bptest(data_test_1)

#Since the p value is 0.19 which is greater than 0.05, 
#we fail to reject the null hypothesis. we state that there is not enough statisticallty significant 
#evidence of heteroskedasticity exist in the data set.



####### The Multicollinearity
vif(data_test_1)
#All the VIF values for the variable are between 1 and 1.6. This indicates that 
#the training data show moderate correlation, which is common for real-life examples, and will not raise a red flag.


######################



######################
#model one: The linear regression
######################
######################
#Step two: Sharking method
  #best subset selection 
  #Lasso regression
#####################
#using the ridge regession to search all the possible linear combination 

#create a new matrix
x <- TTD %>%
  dplyr::select(
    Pclass, Age, SibSp, Parch, Fare, Embarked, Sex_female
  ) %>%
  data.matrix()

#take out the survived data.
live <- TTD$Survived

#The data frame time!
data2 <- data.frame(x, live)

#using regression to search all the possible output
regfit_all <- regsubsets(live ~ ., data = data2, nvmax = 10)
reg_summary <- summary(regfit_all)

# ploting the result. 
plot(regfit_all, scale = "bi ")

########## using the sharking method to determined the optimal output for the number of variable
# 
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

#####################

#According to the best subselection, I should choose two group as the following variable
# the first group
  #Pclass, Age, sibsp, Sex_female
# the second group
  #Pclass, Age, sibsp, Embarked, Sex_female 


########## The Ridge Regression Time ##########

# using the cross vaildation to find the optimal lambda
ridge.cv <- cv.glmnet(x, live, alpha = 0, nfolds = 5)

# imput the corss vaildation
coef_ridge <- as.vector(coef(ridge.cv, s = "lambda.min")[-1]) 

#OLS time
ols_mod <- lm(live ~ ., data = data2)
coef_ols <- coef(ols_mod)[-1]  

# Create a data frame for comparison
variable_names <- names(coef_ols)


#put all the result together.
performence_table <- data.frame(
  Variable = variable_names,
  OLS = coef_ols,
  Ridge = coef_ridge)

# Display the comparison table sorted by Ridge coefficients
performence_table_ordered <- performence_table[order(performence_table$Ridge), ]

# Display the sorted comparison table
print(performence_table_ordered)



#The lasso regression does not sharking enough meanslasso regression found
#all the variable are actually have some degree of importantence. 


#according the lasso regression we should use the 
  #Sex and Pclass

############ OLS model time
# the first group
#Pclass, Age, sibsp, Sex_female
# the second group
#Pclass, Age, sibsp, Embarked, Sex_female 


#####################
#model compare
#####################
#using the 2/3 of training data to traning and using 1/3 of the training data to test
#which group has the lowest BIC which group win!

#TTD

# Split the data

set.seed(123455667)
training_indices <- sample(1:nrow(TTD), size = 9/10 * nrow(TTD))
training_data <- TTD[training_indices, ]
testing_data <- TTD[-training_indices, ]

# Fit OLS models for each group on the training data
model_group1 <- lm(Survived ~ Pclass + Sex_female + Age + SibSp, data = training_data)
model_group2 <- lm(Survived ~ Pclass + Sex_female + Age + SibSp + Embarked, data =training_data)

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


















#####################
#prediction time!
#####################
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

The_final_test <- cbind(test_data_set_without_age_gap$PassengerId, test_data_set_without_age_gap$Predicted_Survived )

# Save to CSV
write.csv(The_final_test, "The_final_test.csv", row.names = FALSE)






#####################


#####################
#model two: The Logistic regression(logit model)
#####################

###This is the second model I want to try
#why logistic regression?
  #because it directly model the probability of the outcome as a function of the predictors.
  #Its the base line for the binary  classification problem

#now my question is, will it proform better than the linear regression? 

rm( list = ls())

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

################ The age gap. 
#I notice there are some age part are empty so i decided to use mean of age to fill the gap. 
#but before that I just need to check the distribution of age between the survived and died



#calcuate the average of all the age 
The_average_age <- mean(Titanic_train_raw$Age, na.rm = TRUE)
#print(The_average_age)


######### fill the age gap
#they are close enough, now fill the age gap. 
#fill the "age gap"
Titanic_train_raw$Age[is.na(Titanic_train_raw$Age)] <- The_average_age
Titanic_train_without_age_gap <- Titanic_train_raw
########The data combination#####
#delete the 2 line in the embarked 
Titanic_train <- Titanic_train_without_age_gap[!is.na(Titanic_train_without_age_gap$Embarked), ]
Titanic_train_cleaned <- na.omit(Titanic_train_without_age_gap)

#delete all the data that including at least one NA value
  #Change all the gender to the dummy variable 
    #male = 1 female equal to zero 
    TTD <- dummy_cols(Titanic_train_cleaned, select_columns = "Sex", remove_selected_columns = TRUE)

#####################
#2. sharnking method
#####################
#this part is empty because I already using the l1 Ridge regressiond and L2 lasso regerssion
#did the sharning. suprisely it wokring for the logist regression too.
    
####################
#3.The model comparsing
    #TTD
    
#cut the date in the 2 part
    #select 90% for the training
    #select 10% for the testing

#calculate how many row in the TTD data set 
cut_data<- sample(1:nrow(TTD), size = 9/10* nrow(TTD), replace = FALSE)
  
#select the 90% of the data set for the training
training_set <- TTD[cut_data, ]
#rest of 10% for the test 
testing_set <- TTD[-cut_data, ]
    
# Fit the logistic regression model
model1 <- glm(Survived ~ Pclass + Sex_female + Age + SibSp, data = training_set)
model2 <- glm(Survived ~ Pclass + Sex_female + Age + SibSp + Embarked, data =training_set)
    

# Making predictions on the test set for each model
model1_pred <- predict(model1, newdata = testing_set, type = "response")
model2_pred<- predict(model2, newdata = testing_set, type = "response")
    
# Actual values
test_time <- testing_set$Survived
    
    
# Calculate MSE, RMSE, and MAE for each group
  #for the model_1
  model1_MSE <- mean((test_time -model1_pred)^2)
  model1_RMSE <- sqrt(mean((test_time -model1_pred)^2))
  model1_MAE <- mean(abs(test_time -model1_pred))
  
  #for the model_2
  model2_MSE <- mean((test_time -model2_pred)^2)
  model2_RMSE <- sqrt(mean((test_time -model2_pred)^2))
  model2_MAE <- mean(abs(test_time -model2_pred))
  
#The final result 
  The_final_table_logic <- data.frame (
    test = c("MSE", "RMSE","MAE"),
    model_1 = c( model1_MSE,model1_RMSE, model1_MAE),
    model_2 = c( model2_MSE,model2_RMSE, model2_MAE))

#print the final result
    print(The_final_table_logic)
    
#The model 1 and model 2 profermence are so close but
    #model 2 a bit better in every part
  

#The winning linear regression model2
  
########The prediction time

test_data_set <- read_csv("Titanic data/test.csv", 
                              col_types = cols(Name = col_skip(), Parch = col_skip(), 
                                               Ticket = col_skip(), Fare = col_skip(), 
                                               Cabin = col_skip()))

# Fill in missing Age values with the mean Age, ensuring NA values are handled
test_data_set_without_age_gap <- test_data_set %>%
mutate(Age = ifelse(is.na(Age), mean(Age, na.rm = TRUE), Age))
    
#Transfer the dummy varaible
test_data_ultra<- dummy_cols(test_data_set_without_age_gap, select_columns = "Sex", remove_selected_columns = TRUE)

# Prediction
prediction <- predict(model2, newdata = test_data_ultra, type = "response")

prediction_cleaned <- ifelse(prediction > 0.5, 1, 0)

# Add Predicted_Survived to the dataset
test_data_set_without_age_gap$Survived <- prediction_cleaned

The_final_test_logit_model <- data_frame(PassengerId =test_data_set_without_age_gap$PassengerId, 
                                         Survived =test_data_set_without_age_gap$Survived )
 
 # output
write.csv(The_final_test_logit_model, "The_final_test_logit_model.csv", row.names = FALSE)

#0.76794 for the logit model. 

#############
#model 3: the randomForest
############
############data import and cleaning
#clean the enviroment and import the date
rm( list = ls())

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

################ The age gap. 
#I notice there are some age part are empty so i decided to use mean of age to fill the gap. 
#but before that I just need to check the distribution of age between the survived and died



#calcuate the average of all the age 
The_average_age <- mean(Titanic_train_raw$Age, na.rm = TRUE)
#print(The_average_age)


######### fill the age gap
#they are close enough, now fill the age gap. 
#fill the "age gap"
Titanic_train_raw$Age[is.na(Titanic_train_raw$Age)] <- The_average_age
Titanic_train_without_age_gap <- Titanic_train_raw
########The data combination#####
#delete the 2 line in the embarked 
Titanic_train <- Titanic_train_without_age_gap[!is.na(Titanic_train_without_age_gap$Embarked), ]
Titanic_train_cleaned <- na.omit(Titanic_train_without_age_gap)

#delete all the data that including at least one NA value
#Change all the gender to the dummy variable 
#male = 1 female equal to zero 
TTD <- dummy_cols(Titanic_train_cleaned, select_columns = "Sex", remove_selected_columns = TRUE)

#############

############ 
library(randomForest)
library(caret)
############
TTD$Survived <- as.factor(TTD$Survived) 

###########

# Prepare your data (TTD) and the trainControl for cross-validation
control <- trainControl(method="cv", number=10, search="grid", allowParallel = TRUE)

# Define a sequence of ntree values to test
ntreeGrid <- expand.grid(mtry=1:10)

# Train the model across the ntree range
set.seed(12345)
# Note: mtry might need to be set or explored as well; here we use the default sqrt(number of predictors)
model_test <- train(Survived ~ ., data=TTD, method="rf", trControl=control, tuneGrid=ntreeGrid)

print(model_test)
# the optimual mtry is 5


##########

##########

#calculate how many row in the TTD data set 
cut_data<- sample(1:nrow(TTD), size = 9/10* nrow(TTD), replace = FALSE)

#select the 90% of the data set for the training
training_set <- TTD[cut_data, ]
#rest of 10% for the test 
testing_set <- TTD[-cut_data, ]


rf_model <- randomForest(Survived ~ ., data = training_set, 
                         ntree = 10000, 
                         mtry = 3, 
                         nodesize = 1, 
                         importance = TRUE)
print(rf_model)

predictions <- predict(rf_model, newdata = testing_set)

# Assuming 'Survived' is a factor and predictions are made accordingly
actual <- testing_set$Survived
conf_matrix <- table(Predicted = predictions, Actual = actual)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", accuracy))

####

setwd("/Users/tie/Documents/GitHub/The-data-analysis-job--")

# Import test dataset
test_data_set <- read_csv("Titanic data/test.csv",col_types = cols( Name = col_skip(), Ticket = col_skip(), 
                                                                    Cabin = col_skip())
                          )

# Replace missing Age values with the mean Age from the training set
mean_age <- mean(training_set$Age, na.rm = TRUE)
test_data_set$Age <- ifelse(is.na(test_data_set$Age), mean_age, test_data_set$Age)

# Convert the 'Sex' column into dummy variables and remove the original 'Sex' column
test_data_ultra <- dummy_cols(test_data_set, select_columns = "Sex", remove_selected_columns = TRUE)

# Assuming rf_model is your trained Random Forest model
# and test_data_ultra is your prepared test dataset that includes PassengerId

# 假设你已经完成了预测
predictions <- predict(rf_model, newdata = test_data_ultra, type = "response")

# 将预测结果转换为二进制格式（0和1），如果需要的话
binary_predictions <- as.integer(predictions) - 1

# 将二进制预测结果添加到测试数据集中
test_data_ultra$Survived <- binary_predictions

# 准备最终的输出数据框，确保包含PassengerId和Survived列
final_output_evil <- data.frame(PassengerId = test_data_ultra$PassengerId, Survived = test_data_ultra$Survived)

# 将输出保存到CSV文件中
write.csv(final_output_evil, "final_predictions_rf.csv", row.names = FALSE)

