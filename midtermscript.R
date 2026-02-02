############################################################
# DONOR RESPONSE PREDICTION
# Logistic Regression, LDA, QDA, KNN
# Uses: MASS, class, pROC
# File: ~/Downloads/donorsfile.csv  (change path if needed)
############################################################

set.seed(123)

# ---------------------------
# 0) Packages
# ---------------------------
if (!requireNamespace("MASS", quietly = TRUE)) install.packages("MASS")
if (!requireNamespace("class", quietly = TRUE)) install.packages("class")
if (!requireNamespace("pROC", quietly = TRUE)) install.packages("pROC")

library(MASS)   # lda, qda
library(class)  # knn
library(pROC)   # AUC

# ---------------------------
# 1) Load data  (EDIT PATH)
# ---------------------------
donors <- read.csv("~/Downloads/donorsfile.csv", stringsAsFactors = FALSE)

# ---------------------------
# 2) Target + variable prep
# ---------------------------

# Target -> factor(FALSE, TRUE)
y_raw <- donors$respondedMailing
if (is.numeric(y_raw)) {
  donors$respondedMailing <- factor(y_raw == 1, levels = c(FALSE, TRUE))
} else {
  y_chr <- tolower(trimws(as.character(y_raw)))
  donors$respondedMailing <- factor(y_chr %in% c("true","t","yes","y","1"),
                                    levels = c(FALSE, TRUE))
}

# isHomeowner: TRUE/NA only -> Homeowner/Unknown/Other
donors$isHomeowner <- ifelse(is.na(donors$isHomeowner), "Unknown",
                             ifelse(donors$isHomeowner %in% c(TRUE,"TRUE","true",1,"1"),
                                    "Homeowner", "Other"))
donors$isHomeowner <- factor(donors$isHomeowner)

# Program flags -> Yes/No/Unknown
make_yesno_unknown <- function(x) {
  x2 <- ifelse(is.na(x), "Unknown", ifelse(as.logical(x), "Yes", "No"))
  factor(x2)
}
donors$inHouseDonor        <- make_yesno_unknown(donors$inHouseDonor)
donors$plannedGivingDonor  <- make_yesno_unknown(donors$plannedGivingDonor)
donors$sweepstakesDonor    <- make_yesno_unknown(donors$sweepstakesDonor)
donors$P3Donor             <- make_yesno_unknown(donors$P3Donor)

# Categorical predictors -> factor
donors$state               <- factor(donors$state)
donors$urbanicity          <- factor(donors$urbanicity)
donors$socioEconomicStatus <- factor(donors$socioEconomicStatus)
donors$gender              <- factor(donors$gender)

# ---------------------------
# 3) Missing values
# ---------------------------

# numeric -> median
num_cols <- names(donors)[sapply(donors, is.numeric)]
for (c in num_cols) donors[[c]][is.na(donors[[c]])] <- median(donors[[c]], na.rm = TRUE)

# factor/character -> "Unknown"
fac_cols <- setdiff(names(donors)[sapply(donors, function(x) is.factor(x) || is.character(x))],
                    "respondedMailing")
for (c in fac_cols) {
  donors[[c]] <- as.character(donors[[c]])
  donors[[c]][is.na(donors[[c]])] <- "Unknown"
  donors[[c]] <- factor(donors[[c]])
}

# ---------------------------
# 4) CRITICAL FIX: collapse rare states BEFORE split
#    prevents "new levels" errors in predict()
# ---------------------------
state_counts <- table(donors$state)
rare_states <- names(state_counts[state_counts < 50])   # adjust cutoff if you want
donors$state <- as.character(donors$state)
donors$state[donors$state %in% rare_states] <- "Other"
donors$state <- factor(donors$state)

# ---------------------------
# 5) Train/Test split (70/30 stratified)
# ---------------------------
idx_true  <- which(donors$respondedMailing == TRUE)
idx_false <- which(donors$respondedMailing == FALSE)

train_idx <- c(
  sample(idx_true,  size = floor(0.7 * length(idx_true))),
  sample(idx_false, size = floor(0.7 * length(idx_false)))
)
train_idx <- sort(train_idx)

train <- donors[train_idx, ]
test  <- donors[-train_idx, ]

# Align factor levels of test to train (extra safety)
factor_cols <- names(train)[sapply(train, is.factor)]
factor_cols <- setdiff(factor_cols, "respondedMailing")
for (col in factor_cols) {
  test[[col]] <- factor(as.character(test[[col]]), levels = levels(train[[col]]))
}

# ---------------------------
# Helper: confusion matrix + accuracy + AUC
# ---------------------------
eval_binary <- function(y_true, prob_true, thr = 0.5) {
  pred <- factor(prob_true >= thr, levels = c(FALSE, TRUE))
  cm <- table(Pred = pred, Actual = y_true)
  acc <- mean(pred == y_true)
  auc <- as.numeric(pROC::auc(y_true, prob_true))
  list(cm = cm, acc = acc, auc = auc, pred = pred)
}

# ============================================================
# 6) LOGISTIC REGRESSION
# ============================================================
logit_fit  <- glm(respondedMailing ~ ., data = train, family = binomial)
logit_prob <- predict(logit_fit, newdata = test, type = "response")

# choose threshold (0.5 default; you used 0.05 earlier to catch more TRUEs)
logit_eval_05 <- eval_binary(test$respondedMailing, logit_prob, thr = 0.05)
cat("\n===== Logistic Regression (thr=0.05) =====\n")
print(logit_eval_05$cm)
cat("Accuracy:", round(logit_eval_05$acc, 4), "\n")
cat("AUC:", round(logit_eval_05$auc, 4), "\n")

# ============================================================
# 7) LDA (fixed: use model.matrix + drop constant-within-class)
# ============================================================

# Build design matrices (one-hot)
X_train <- model.matrix(respondedMailing ~ ., data = train)[, -1, drop = FALSE]
X_test  <- model.matrix(respondedMailing ~ ., data = test)[, -1, drop = FALSE]

# align columns
common <- intersect(colnames(X_train), colnames(X_test))
X_train <- X_train[, common, drop = FALSE]
X_test  <- X_test[, common, drop = FALSE]

y_train <- train$respondedMailing
y_test  <- test$respondedMailing

# Drop predictors that are constant within either class (LDA requirement)
keep <- apply(X_train, 2, function(v) {
  ok_false <- length(unique(v[y_train == FALSE])) > 1
  ok_true  <- length(unique(v[y_train == TRUE]))  > 1
  ok_false && ok_true
})
X_train2 <- X_train[, keep, drop = FALSE]
X_test2  <- X_test[, keep, drop = FALSE]

lda_fit <- MASS::lda(x = X_train2, grouping = y_train)  # avoids formula constant-within-group issues
lda_out <- predict(lda_fit, newdata = X_test2)

lda_prob <- lda_out$posterior[, "TRUE"]
lda_pred <- lda_out$class

lda_cm  <- table(Pred = lda_pred, Actual = y_test)
lda_acc <- mean(lda_pred == y_test)
lda_auc <- as.numeric(pROC::auc(y_test, lda_prob))

cat("\n===== LDA =====\n")
print(lda_cm)
cat("Accuracy:", round(lda_acc, 4), "\n")
cat("AUC:", round(lda_auc, 4), "\n")

# Optional: use threshold like you did (0.05)
lda_pred_05 <- factor(lda_prob >= 0.05, levels = c(FALSE, TRUE))
cat("\n===== LDA (thr=0.05) =====\n")
print(table(Pred = lda_pred_05, Actual = y_test))
cat("Accuracy:", round(mean(lda_pred_05 == y_test), 4), "\n")

# ============================================================
# 8) QDA (fixed: drop constant-within-class + drop collinear columns)
# ============================================================

# QDA can fail with rank deficiency (singular covariance).
# Fix: keep only linearly independent columns using QR.
qr_keep <- qr(X_train2)$pivot[seq_len(qr(X_train2)$rank)]
X_train3 <- X_train2[, qr_keep, drop = FALSE]
X_test3  <- X_test2[, qr_keep, drop = FALSE]

qda_fit <- MASS::qda(x = X_train3, grouping = y_train)
qda_out <- predict(qda_fit, newdata = X_test3)

qda_prob <- qda_out$posterior[, "TRUE"]
qda_pred <- qda_out$class

qda_cm  <- table(Pred = qda_pred, Actual = y_test)
qda_acc <- mean(qda_pred == y_test)
qda_auc <- as.numeric(pROC::auc(y_test, qda_prob))

cat("\n===== QDA =====\n")
print(qda_cm)
cat("Accuracy:", round(qda_acc, 4), "\n")
cat("AUC:", round(qda_auc, 4), "\n")

############################################################
# 9) KNN (FAST VERSION) - numeric only (drops state)
############################################################

# y labels
y_train <- train$respondedMailing
y_test  <- test$respondedMailing

# Use ONLY numeric predictors (KNN works best here)
num_vars <- names(train)[sapply(train, is.numeric)]
num_vars <- setdiff(num_vars, "respondedMailing")  # just in case

X_train_knn <- as.matrix(train[, num_vars])
X_test_knn  <- as.matrix(test[,  num_vars])

# scale (train stats)
mu  <- colMeans(X_train_knn)
sdv <- apply(X_train_knn, 2, sd)
sdv[sdv == 0] <- 1

X_train_sc <- scale(X_train_knn, center = mu, scale = sdv)
X_test_sc  <- scale(X_test_knn,  center = mu, scale = sdv)

# Try small k values first (fast)
k_grid <- c(1, 3, 5, 7, 9, 15, 25)
knn_res <- data.frame(k = k_grid, accuracy = NA_real_)

for (i in seq_along(k_grid)) {
  k <- k_grid[i]
  pred <- class::knn(train = X_train_sc, test = X_test_sc, cl = y_train, k = k)
  knn_res$accuracy[i] <- mean(pred == y_test)
}

knn_res
best_k <- knn_res$k[which.max(knn_res$accuracy)]
best_k

# Confusion matrix for best k
knn_pred <- class::knn(train = X_train_sc, test = X_test_sc, cl = y_train, k = best_k)
table(Pred = knn_pred, Actual = y_test)
mean(knn_pred == y_test)
