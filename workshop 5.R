install.packages(c("broom", "car", "pROC", "performance", "marginaleffects", "caret", "tidyverse"))

library(broom)
library(car)
library(pROC)
library(performance)
library(marginaleffects)
library(caret)
library(tidyverse)

# Web address of the dataset on GitHub
url <- "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
# Read the data and set up the categorical variables
telco <- read_csv(url, show_col_types = FALSE) %>%
  mutate(
    Churn = factor(Churn, levels = c("No", "Yes")),
    Contract = factor(Contract, levels = c("Month-to-month", "One year", "Two year")),
    SeniorCitizen = factor(SeniorCitizen, levels = c(0, 1), labels = c("No", "Yes")),
    PaperlessBilling = factor(PaperlessBilling, levels = c("No", "Yes"))
  )

# Examine the structure of the dataset
glimpse(telco)
# Obtain descriptive statistics
summary(telco)
# Count the number of observations
nrow(telco)
# Count and percentage of customers who churned
table(telco$Churn)
prop.table(table(telco$Churn))
# Check for missing values
colSums(is.na(telco))
# Check for duplicate customer IDs
sum(duplicated(telco$customerID))
# Churn rate by contract type
telco %>%
  group_by(Contract) %>%
  summarise(customers = n(), churn_rate = mean(Churn == "Yes"))

# Inspect the customers with missing TotalCharges
telco %>%
  filter(is.na(TotalCharges)) %>%
  select(customerID, tenure, MonthlyCharges, TotalCharges)
# Remove them
telco <- telco %>% drop_na(TotalCharges)
nrow(telco)

# Boxplot of tenure by churn status
ggplot(telco, aes(x = Churn, y = tenure)) +
  geom_boxplot() +
  labs(title = "Tenure by Churn Status", x = "Churned", y = "Tenure (months)") +
  theme_minimal()
# Fitted logistic curve
telco %>%
  mutate(churn_num = as.numeric(Churn == "Yes")) %>%
  ggplot(aes(x = tenure, y = churn_num)) +
  geom_jitter(height = 0.05, alpha = 0.1) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  labs(title = "Probability of Churn by Tenure", x = "Tenure (months)", y = "Probability of churn") +
  theme_minimal()
# Fit the simple logistic regression model
simple <- glm(Churn ~ tenure, data = telco, family = "binomial")
# Examine the results
summary(simple)

# Odds ratios with 95% confidence intervals
exp(cbind(OR = coef(simple), confint.default(simple)))

# Create new customers
new_customers <- data.frame(tenure = c(1, 12, 36, 72))
# Predict the probability of churn
predict(simple, newdata = new_customers, type = "response")

# Fit the multiple logistic regression model
multiple <- glm(Churn ~ tenure + MonthlyCharges + Contract + SeniorCitizen + PaperlessBilling,
                data = telco, family = "binomial")
# Examine the results
summary(multiple)
# Odds ratios with 95% confidence intervals
round(exp(cbind(OR = coef(multiple), confint.default(multiple))), 3)

# Fit a model with no predictors (the null model)
null_model <- glm(Churn ~ 1, data = telco, family = "binomial")
# Likelihood ratio (chi-square) test
anova(null_model, multiple, test = "Chisq")
# Pseudo R-squared
r2_mcfadden(multiple)
r2_nagelkerke(multiple)

# Compare the two models
anova(simple, multiple, test = "Chisq")
AIC(simple, multiple)

# Average marginal effects
avg_slopes(multiple)

# VIF for the multiple model
vif(multiple)
# Extended model including TotalCharges
extended <- glm(Churn ~ tenure + MonthlyCharges + TotalCharges + Contract + SeniorCitizen + PaperlessBilling,
                data = telco, family = "binomial")
vif(extended)
# Correlation between tenure and TotalCharges
cor(telco$tenure, telco$TotalCharges)

# Predicted probabilities
telco$prob <- predict(multiple, type = "response")
# Classify using a 0.5 threshold
telco$pred <- factor(ifelse(telco$prob > 0.5, "Yes", "No"), levels = c("No", "Yes"))
# Confusion matrix
cm <- table(Actual = telco$Churn, Predicted = telco$pred)
cm
# Accuracy, sensitivity and specificity
accuracy <- sum(diag(cm)) / sum(cm)
sensitivity <- cm["Yes", "Yes"] / sum(cm["Yes", ])
specificity <- cm["No", "No"] / sum(cm["No", ])
round(c(accuracy = accuracy, sensitivity = sensitivity, specificity = specificity), 3)

# ROC curve
roc_obj <- roc(telco$Churn, telco$prob)
plot(roc_obj, print.thres = "best", print.auc = TRUE)
# Area under the curve
auc(roc_obj)
# Best threshold using Youden's J
best <- coords(roc_obj, "best", best.method = "youden")
best
# Classify using the best threshold
telco$pred_best <- factor(ifelse(telco$prob > best$threshold, "Yes", "No"), levels = c("No", "Yes"))
table(Actual = telco$Churn, Predicted = telco$pred_best)

# Split the data: 70% training, 30% test
set.seed(42)
train_rows <- createDataPartition(telco$Churn, p = 0.7, list = FALSE)
train <- telco[train_rows, ]
test <- telco[-train_rows, ]
# Check the churn rate in each set
prop.table(table(train$Churn))
prop.table(table(test$Churn))
# Fit the model on the training data only
model_train <- glm(Churn ~ tenure + MonthlyCharges + Contract + SeniorCitizen + PaperlessBilling,
                   data = train, family = "binomial")
# Predict on both sets
train$prob <- predict(model_train, type = "response")
test$prob <- predict(model_train, newdata = test, type = "response")
# Compare AUC on training and test data
roc_train <- roc(train$Churn, train$prob)
roc_test <- roc(test$Churn, test$prob)
auc(roc_train)
auc(roc_test)
# Choose the threshold on the training data, then apply it to the test data
best_train <- coords(roc_train, "best", best.method = "youden")
test$pred <- factor(ifelse(test$prob > best_train$threshold, "Yes", "No"), levels = c("No", "Yes"))
table(Actual = test$Churn, Predicted = test$pred)

# Thresholds to compare
thresholds <- seq(0.10, 0.90, by = 0.05)
# Total cost of errors for each threshold
cost_table <- tibble(threshold = thresholds) %>%
  rowwise() %>%
  mutate(
    false_negatives = sum(test$prob <= threshold & test$Churn == "Yes"),
    false_positives = sum(test$prob > threshold & test$Churn == "No"),
    total_cost = false_negatives * 500 + false_positives * 100
  ) %>%
  ungroup()
cost_table
# Threshold with the lowest total cost
cost_table %>% slice_min(total_cost)
# Plot total cost against threshold
ggplot(cost_table, aes(x = threshold, y = total_cost)) +
  geom_line() +
  geom_point() +
  labs(title = "Total Cost of Errors by Threshold", x = "Threshold", y = "Total cost ($)") +
  theme_minimal()


