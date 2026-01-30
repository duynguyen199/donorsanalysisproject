
install.packages("tidyverse")
install.packages(c("caret","dplyr","MASS","class","pROC","janitor"),
                 repos = "https://cloud.r-project.org")

library(tidyverse)
library(caret)
library(MASS)
library(class)
library(pROC)
library(janitor)

donors <- read.csv("~/Downloads/donorsfile.csv")

names(donors)


library(MASS)   # LDA/QDA
library(class)  # KNN
library(pROC)   # AUC

set.seed(123)

donors <- read.csv("~/Downloads/donorsfile.csv", stringsAsFactors = FALSE)


# convert and edit data to FALSE/TRUE
y_raw <- donors$respondedMailing
if (is.numeric(y_raw)) {
  donors$respondedMailing <- factor(y_raw == 1, levels = c(FALSE, TRUE))
} else {
  y_chr <- tolower(trimws(as.character(y_raw)))
  donors$respondedMailing <- factor(y_chr %in% c("true","t","yes","y","1"),
                                    levels = c(FALSE, TRUE))
}

donors$isHomeowner <- ifelse(is.na(donors$isHomeowner), "Unknown",
                             ifelse(donors$isHomeowner %in% c(TRUE,"TRUE","true",1,"1"),
                                    "Homeowner", "Other"))
donors$isHomeowner <- factor(donors$isHomeowner)

# Convert factors to Yes/No/Unknown.
make_yesno_unknown <- function(x) {
  x2 <- ifelse(is.na(x), "Unknown", ifelse(as.logical(x), "Yes", "No"))
  factor(x2)
}
donors$inHouseDonor        <- make_yesno_unknown(donors$inHouseDonor)
donors$plannedGivingDonor  <- make_yesno_unknown(donors$plannedGivingDonor)
donors$sweepstakesDonor    <- make_yesno_unknown(donors$sweepstakesDonor)
donors$P3Donor             <- make_yesno_unknown(donors$P3Donor)

# Convert any ategorical predictors to factor for better using
donors$state               <- factor(donors$state)
donors$urbanicity          <- factor(donors$urbanicity)
donors$socioEconomicStatus <- factor(donors$socioEconomicStatus)
donors$gender              <- factor(donors$gender)



#Handle missing values to median
num_cols <- names(donors)[sapply(donors, is.numeric)]
for (c in num_cols) {
  donors[[c]][is.na(donors[[c]])] <- median(donors[[c]], na.rm = TRUE)
}

# Convert missing values to Unknown
fac_cols <- setdiff(names(donors)[sapply(donors, function(x) is.factor(x) || is.character(x))],
                    "respondedMailing")
for (c in fac_cols) {
  donors[[c]] <- as.character(donors[[c]])
  donors[[c]][is.na(donors[[c]])] <- "Unknown"
  donors[[c]] <- factor(donors[[c]])
}

# Using 70/30 train/test split
idx_true  <- which(donors$respondedMailing == TRUE)
idx_false <- which(donors$respondedMailing == FALSE)

train_idx <- c(
  sample(idx_true,  size = floor(0.7 * length(idx_true))),
  sample(idx_false, size = floor(0.7 * length(idx_false)))
)
train_idx <- sort(train_idx)
test_idx  <- setdiff(seq_len(nrow(donors)), train_idx)

train <- donors[train_idx, ]
test  <- donors[test_idx, ]


# Handle some label factor to Other if not present
fix_unseen_levels <- function(train_df, test_df, response = "respondedMailing") {
  facs <- names(train_df)[sapply(train_df, is.factor)]
  facs <- setdiff(facs, response)
  
  for (col in facs) {
    tr <- as.character(train_df[[col]])
    te <- as.character(test_df[[col]])
    
    if (!("Other" %in% tr)) tr_levels <- c(sort(unique(tr)), "Other") else tr_levels <- sort(unique(tr))
    
    te[!(te %in% tr_levels)] <- "Other"
    tr[!(tr %in% tr_levels)] <- "Other"
    
    train_df[[col]] <- factor(tr, levels = tr_levels)
    test_df[[col]]  <- factor(te, levels = tr_levels)
  }
  
  list(train = train_df, test = test_df)
}

fixed <- fix_unseen_levels(train, test, response = "respondedMailing")
train <- fixed$train
test  <- fixed$test


# Logistic Regression
logit_fit <- glm(respondedMailing ~ ., data = train, family = binomial)

logit_prob <- predict(logit_fit, newdata = test, type = "response")
logit_pred <- factor(logit_prob >= 0.05, levels = c(FALSE, TRUE))

logit_cm  <- table(Pred = logit_pred, Actual = test$respondedMailing)
logit_acc <- mean(logit_pred == test$respondedMailing)
logit_auc <- as.numeric(pROC::auc(test$respondedMailing, logit_prob))

cat("\n===== Logistic Regression =====\n")
print(logit_cm)
cat("Accuracy:", round(logit_acc, 4), "\n")
cat("AUC:", round(logit_auc, 4), "\n")


# LDA
# This will show the predictor names corresponding to the error positions
bad_pos <- c(14, 16, 18, 20, 77, 83, 87, 89, 93)

# Build the same design matrix LDA uses (this is what those positions refer to)
X_mm <- model.matrix(respondedMailing ~ ., data = train_clean)

# Drop intercept column
X_names <- colnames(X_mm)[-1]

bad_names <- X_names[bad_pos]
bad_names
# Response
y_train <- train_clean$respondedMailing
y_test  <- test_clean$respondedMailing

# Design matrices (one-hot encoding)
X_train <- model.matrix(respondedMailing ~ ., data = train_clean)[, -1, drop = FALSE]
X_test  <- model.matrix(respondedMailing ~ ., data = test_clean)[, -1, drop = FALSE]

# Align columns
common <- intersect(colnames(X_train), colnames(X_test))
X_train <- X_train[, common, drop = FALSE]
X_test  <- X_test[, common, drop = FALSE]

# Remove the "bad" columns (from Step A) IF they exist
bad_present <- intersect(bad_names, colnames(X_train))
X_train2 <- X_train[, setdiff(colnames(X_train), bad_present), drop = FALSE]
X_test2  <- X_test[,  setdiff(colnames(X_test),  bad_present), drop = FALSE]

# Fit LDA using x/y interface
lda_fit <- MASS::lda(x = X_train2, grouping = y_train)

# Predict
lda_out  <- predict(lda_fit, newdata = X_test2)
lda_pred <- lda_out$class
lda_prob <- lda_out$posterior[, "TRUE"]

# Evaluate
lda_cm  <- table(Pred = lda_pred, Actual = y_test)
lda_acc <- mean(lda_pred == y_test)
lda_auc <- as.numeric(pROC::auc(y_test, lda_prob))
lda_pred_05 <- factor(lda_prob >= 0.05, levels = c(FALSE, TRUE))
table(Pred = lda_pred_05, Actual = y_test)
mean(lda_pred_05 == y_test)

cat("\n===== LDA (fixed) =====\n")
print(lda_cm)
cat("Accuracy:", round(lda_acc, 4), "\n")
cat("AUC:", round(lda_auc, 4), "\n")


# 8) QDA

y_train <- train$respondedMailing
y_test  <- test$respondedMailing

X_train <- model.matrix(respondedMailing ~ ., data = train)[, -1, drop = FALSE]
X_test  <- model.matrix(respondedMailing ~ ., data = test)[, -1, drop = FALSE]

# Align columns
common <- intersect(colnames(X_train), colnames(X_test))
X_train <- X_train[, common, drop = FALSE]
X_test  <- X_test[, common, drop = FALSE]
drop_const_within <- function(X, y) {
  keep <- apply(X, 2, function(col) {
    length(unique(col[y == FALSE])) > 1 && length(unique(col[y == TRUE])) > 1
  })
  X[, keep, drop = FALSE]
}

X_train2 <- drop_const_within(X_train, y_train)
X_test2  <- X_test[, colnames(X_train2), drop = FALSE]
qr_keep <- function(X) {
  q <- qr(X)
  X[, q$pivot[seq_len(q$rank)], drop = FALSE]
}

X_train3 <- qr_keep(X_train2)
X_test3  <- X_test2[, colnames(X_train3), drop = FALSE]
qda_fit <- MASS::qda(x = X_train3, grouping = y_train)

qda_out  <- predict(qda_fit, newdata = X_test3)
qda_pred <- qda_out$class
qda_prob <- qda_out$posterior[, "TRUE"]

qda_cm  <- table(Pred = qda_pred, Actual = y_test)
qda_acc <- mean(qda_pred == y_test)
qda_auc <- as.numeric(pROC::auc(y_test, qda_prob))

cat("\n===== QDA (fixed) =====\n")
print(qda_cm)
cat("Accuracy:", round(qda_acc, 4), "\n")
cat("AUC:", round(qda_auc, 4), "\n")


# 9) KNN (one-hot encode + scale; try K grid)

x_train <- model.matrix(respondedMailing ~ ., data = train)[, -1, drop = FALSE]
x_test  <- model.matrix(respondedMailing ~ ., data = test)[, -1, drop = FALSE]

# Align columns safely
common_cols <- intersect(colnames(x_train), colnames(x_test))
x_train <- x_train[, common_cols, drop = FALSE]
x_test  <- x_test[, common_cols, drop = FALSE]

# Scale using training statistics
mu  <- colMeans(x_train)
sdv <- apply(x_train, 2, sd)
sdv[sdv == 0] <- 1

x_train_sc <- scale(x_train, center = mu, scale = sdv)
x_test_sc  <- scale(x_test,  center = mu, scale = sdv)

y_train <- train$respondedMailing
y_test  <- test$respondedMailing

k_grid <- c(1, 3, 5, 7, 9, 15, 25, 50)
knn_table <- data.frame(k = k_grid, accuracy = NA_real_)

for (i in seq_along(k_grid)) {
  k <- k_grid[i]
  pred <- class::knn(train = x_train_sc, test = x_test_sc, cl = y_train, k = k)
  knn_table$accuracy[i] <- mean(pred == y_test)
}

best_k <- knn_table$k[which.max(knn_table$accuracy)]
knn_best_pred <- class::knn(train = x_train_sc, test = x_test_sc, cl = y_train, k = best_k)

knn_cm  <- table(Pred = knn_best_pred, Actual = y_test)
knn_acc <- mean(knn_best_pred == y_test)

cat("\n===== KNN =====\n")
print(knn_table)
cat("Best k:", best_k, "\n")
print(knn_cm)
cat("Accuracy:", round(knn_acc, 4), "\n")

# ---------------------------
# 10) Model comparison
# ---------------------------
results <- data.frame(
  model = c("Logistic", "LDA", "QDA", paste0("KNN(k=", best_k, ")")),
  accuracy = c(logit_acc, lda_acc, qda_acc, knn_acc),
  auc = c(logit_auc, lda_auc, qda_auc, NA_real_)
)

cat("\n===== Model Comparison =====\n")
print(results[order(-results$accuracy), ])