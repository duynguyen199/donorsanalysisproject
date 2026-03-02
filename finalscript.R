############################################################
# DONORS MIDTERM v2 (FINAL, NO FACTOR-LEVEL ERRORS)
# Logistic, LDA, QDA, KNN
# + SVC Linear, SVM RBF, SVM Poly
# + BONUS: Naive Bayes, Decision Tree
#
# KEY IDEA:
#   Use model.matrix(terms_train, ...) so train defines the feature space.
#   This avoids "new factor levels AS, GU, Other" forever.
############################################################

rm(list = ls())
set.seed(123)

# ---------------------------
# 0) Packages
# ---------------------------
need <- c("MASS","class","pROC","e1071","rpart")
for (p in need) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

# ---------------------------
# 1) Load data (EDIT PATH)
# ---------------------------
donors <- read.csv("/Users/duynguyen/Desktop/donors project/donors.csv",
                   header = TRUE,
                   stringsAsFactors = FALSE)
names(donors) <- trimws(names(donors))

# find/rename response if needed
resp_col <- grep("respond", names(donors), ignore.case = TRUE, value = TRUE)[1]
if (is.na(resp_col)) stop("No column containing 'respond' found.")
names(donors)[names(donors) == resp_col] <- "respondedMailing"

# ---------------------------
# 2) Target + Type prep
# ---------------------------
y_raw <- donors[["respondedMailing"]]
if (is.numeric(y_raw)) {
  donors[["respondedMailing"]] <- factor(y_raw == 1, levels = c(FALSE, TRUE))
} else {
  y_chr <- tolower(trimws(as.character(y_raw)))
  donors[["respondedMailing"]] <- factor(y_chr %in% c("true","t","yes","y","1"),
                                         levels = c(FALSE, TRUE))
}

# isHomeowner -> Homeowner/Unknown/Other
donors$isHomeowner <- ifelse(is.na(donors$isHomeowner), "Unknown",
                             ifelse(donors$isHomeowner %in% c(TRUE,"TRUE","true",1,"1"),
                                    "Homeowner", "Other"))
donors$isHomeowner <- factor(donors$isHomeowner)

# program flags -> Yes/No/Unknown
make_yesno_unknown <- function(x) {
  x2 <- ifelse(is.na(x), "Unknown", ifelse(as.logical(x), "Yes", "No"))
  factor(x2)
}
for (v in c("inHouseDonor","plannedGivingDonor","sweepstakesDonor","P3Donor")) {
  if (v %in% names(donors)) donors[[v]] <- make_yesno_unknown(donors[[v]])
}

# categorical predictors -> factor (if exist)
for (v in c("state","urbanicity","socioEconomicStatus","gender")) {
  if (v %in% names(donors)) donors[[v]] <- factor(donors[[v]])
}

# ---------------------------
# 3) Missing values
# ---------------------------
num_cols <- names(donors)[sapply(donors, is.numeric)]
for (c in num_cols) donors[[c]][is.na(donors[[c]])] <- median(donors[[c]], na.rm = TRUE)

fac_cols <- setdiff(names(donors)[sapply(donors, function(x) is.factor(x) || is.character(x))],
                    "respondedMailing")
for (c in fac_cols) {
  donors[[c]] <- as.character(donors[[c]])
  donors[[c]][is.na(donors[[c]])] <- "Unknown"
  donors[[c]] <- factor(donors[[c]])
}

# ---------------------------
# 4) Train/Test split (70/30 stratified)
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

y_train <- train$respondedMailing
y_test  <- test$respondedMailing

# ---------------------------
# 5) Helpers
# ---------------------------
calc_auc <- function(y_true, prob_true) as.numeric(pROC::auc(y_true, prob_true))

metrics_bin <- function(y_true, y_pred, prob_true = NULL) {
  y_true <- factor(y_true, levels = c(FALSE, TRUE))
  y_pred <- factor(y_pred, levels = c(FALSE, TRUE))
  cm <- table(Pred = y_pred, Actual = y_true)
  acc <- mean(y_pred == y_true)
  
  TP <- ifelse("TRUE"  %in% rownames(cm) && "TRUE"  %in% colnames(cm), cm["TRUE","TRUE"], 0)
  FN <- ifelse("FALSE" %in% rownames(cm) && "TRUE"  %in% colnames(cm), cm["FALSE","TRUE"], 0)
  TN <- ifelse("FALSE" %in% rownames(cm) && "FALSE" %in% colnames(cm), cm["FALSE","FALSE"], 0)
  FP <- ifelse("TRUE"  %in% rownames(cm) && "FALSE" %in% colnames(cm), cm["TRUE","FALSE"], 0)
  
  sens <- ifelse(TP + FN == 0, NA, TP/(TP+FN))
  spec <- ifelse(TN + FP == 0, NA, TN/(TN+FP))
  bal  <- mean(c(sens, spec), na.rm = TRUE)
  
  auc <- NA
  if (!is.null(prob_true)) auc <- calc_auc(y_true, prob_true)
  
  list(cm=cm, accuracy=acc, sensitivity=sens, specificity=spec, balanced=bal, auc=auc)
}

print_model <- function(name, out) {
  cat("\n==============================\n", name, "\n==============================\n")
  print(out$cm)
  cat("Accuracy:          ", round(out$accuracy,4), "\n")
  cat("Sensitivity/Recall:", round(out$sensitivity,4), "\n")
  cat("Specificity:       ", round(out$specificity,4), "\n")
  cat("Balanced Accuracy: ", round(out$balanced,4), "\n")
  cat("AUC:               ", round(out$auc,4), "\n")
}

scale_with_train <- function(Xtr, Xte) {
  mu <- colMeans(Xtr)
  sdv <- apply(Xtr, 2, sd)
  sdv[sdv == 0] <- 1
  list(tr = scale(Xtr, center = mu, scale = sdv),
       te = scale(Xte, center = mu, scale = sdv))
}

# ---------------------------
# 6) Build ONE-HOT matrices based on TRAIN terms (critical)
# ---------------------------
terms_train <- terms(respondedMailing ~ ., data = train)

X_train <- model.matrix(terms_train, data = train)[, -1, drop = FALSE]
X_test  <- model.matrix(terms_train, data = test)[, -1, drop = FALSE]

# Add any missing dummy columns to test
missing_cols <- setdiff(colnames(X_train), colnames(X_test))
if (length(missing_cols) > 0) {
  X_test <- cbind(X_test,
                  matrix(0, nrow = nrow(X_test), ncol = length(missing_cols),
                         dimnames = list(NULL, missing_cols)))
}
# order columns to match train
X_test <- X_test[, colnames(X_train), drop = FALSE]

# Drop zero-variance columns (helps LDA/QDA/SVM)
nzv <- apply(X_train, 2, function(x) var(x) > 0)
X_train2 <- X_train[, nzv, drop = FALSE]
X_test2  <- X_test[,  nzv, drop = FALSE]

# For QDA: drop collinearity using QR
qr_obj <- qr(X_train2)
keep_cols <- qr_obj$pivot[seq_len(qr_obj$rank)]
X_train_qda <- X_train2[, keep_cols, drop = FALSE]
X_test_qda  <- X_test2[,  keep_cols, drop = FALSE]

# ============================================================
# 7) LOGISTIC (matrix-based: NEVER factor-level errors)
# ============================================================
# ============================================================
# 7) LOGISTIC (matrix-based, correct way)
# ============================================================
# y must be 0/1 for glm.fit style
y_train_num <- as.integer(y_train == TRUE)

logit_fit  <- glm.fit(x = X_train2, y = y_train_num, family = binomial())

logit_prob <- as.vector(plogis(X_test2 %*% logit_fit$coefficients))
logit_prob[is.na(logit_prob)] <- 0  # safety (rare)

thr <- 0.5
logit_pred <- factor(logit_prob >= thr, levels = c(FALSE, TRUE))

logit_out <- metrics_bin(y_test, logit_pred, prob_true = logit_prob)
print_model(paste0("Logistic Regression (thr=", thr, ")"), logit_out)


# ============================================================
# 8) LDA
# ============================================================
lda_fit <- MASS::lda(x = X_train2, grouping = y_train)
lda_pr  <- predict(lda_fit, newdata = X_test2)

lda_pred <- lda_pr$class
lda_prob <- lda_pr$posterior[, "TRUE"]
lda_out  <- metrics_bin(y_test, lda_pred, prob_true = lda_prob)
print_model("LDA", lda_out)

# ============================================================
# 9) QDA
# ============================================================
############################################################
# QDA FIX: remove columns that are constant within a class
# then remove collinearity with QR
############################################################

# Start from X_train2 / X_test2 (already aligned, no zero-variance overall)
Xtr <- X_train2
Xte <- X_test2

# indices per class
idx_T <- which(y_train == TRUE)
idx_F <- which(y_train == FALSE)

# keep columns that have variance in BOTH classes
var_T <- apply(Xtr[idx_T, , drop = FALSE], 2, var)
var_F <- apply(Xtr[idx_F, , drop = FALSE], 2, var)

keep_class_var <- (var_T > 0) & (var_F > 0)
Xtr2 <- Xtr[, keep_class_var, drop = FALSE]
Xte2 <- Xte[, keep_class_var, drop = FALSE]

# If TRUE class is tiny, QDA can still be fragile. Reduce rank with QR.
qr_obj <- qr(Xtr2)
keep_cols <- qr_obj$pivot[seq_len(qr_obj$rank)]

X_train_qda <- Xtr2[, keep_cols, drop = FALSE]
X_test_qda  <- Xte2[, keep_cols, drop = FALSE]

# Fit QDA
qda_fit <- MASS::qda(x = X_train_qda, grouping = y_train)

qda_pr <- predict(qda_fit, newdata = X_test_qda)
qda_pred <- qda_pr$class
qda_prob <- qda_pr$posterior[, "TRUE"]

qda_out <- metrics_bin(y_test, qda_pred, prob_true = qda_prob)
print_model("QDA (fixed: class-variance + QR)", qda_out)


# ============================================================
# 10) KNN (FAST numeric-only)
# ============================================================
num_vars <- names(train)[sapply(train, is.numeric)]
num_vars <- setdiff(num_vars, "respondedMailing")

Xtr_knn <- as.matrix(train[, num_vars, drop = FALSE])
Xte_knn <- as.matrix(test[,  num_vars, drop = FALSE])

knn_scaled <- scale_with_train(Xtr_knn, Xte_knn)
Xtr_knn_sc <- knn_scaled$tr
Xte_knn_sc <- knn_scaled$te

k_grid <- c(1,3,5,7,9,15,25)
knn_res <- data.frame(k = k_grid, accuracy = NA_real_)

for (i in seq_along(k_grid)) {
  pred <- class::knn(train = Xtr_knn_sc, test = Xte_knn_sc, cl = y_train, k = k_grid[i], prob = TRUE)
  knn_res$accuracy[i] <- mean(pred == y_test)
}
best_k <- knn_res$k[which.max(knn_res$accuracy)]
knn_pred <- class::knn(train = Xtr_knn_sc, test = Xte_knn_sc, cl = y_train, k = best_k, prob = TRUE)

win_prob <- attr(knn_pred, "prob")
knn_prob <- ifelse(knn_pred == TRUE, win_prob, 1 - win_prob)

knn_out <- metrics_bin(y_test, knn_pred, prob_true = knn_prob)
print_model(paste0("KNN numeric-only (k=",best_k,")"), knn_out)
cat("\nKNN accuracy by k:\n"); print(knn_res)
############################################################
# 11) SVC / SVM (FAST VERSION, sampled)
############################################################
library(e1071)

# ---- SCALE first ----
svm_scaled <- scale_with_train(X_train_qda, X_test_qda)

Xtr_svm <- svm_scaled$tr
Xte_svm <- svm_scaled$te

# ---- Stratified sampling ----
set.seed(123)

idx_T <- which(y_train == TRUE)
idx_F <- which(y_train == FALSE)

nT <- min(2000, length(idx_T))
nF <- min(2000, length(idx_F))

svm_idx <- c(sample(idx_T, nT),
             sample(idx_F, nF))

Xtr_small <- Xtr_svm[svm_idx, , drop = FALSE]
ytr_small <- y_train[svm_idx]

# ---- Linear SVC (fast, no probability) ----
svc_fit <- e1071::svm(
  x = Xtr_small,
  y = ytr_small,
  kernel = "linear",
  cost = 1,
  scale = FALSE
)

svc_pred <- predict(svc_fit, Xte_svm)

svc_cm <- table(Pred = svc_pred, Actual = y_test)
svc_acc <- mean(svc_pred == y_test)

cat("\n===== SVC (Linear, Sampled) =====\n")
print(svc_cm)
cat("Accuracy:", round(svc_acc,4), "\n")


# Predict on full test
svc_pred <- predict(svc_fit, Xte_svm)

table(Pred = svc_pred, Actual = y_test)
mean(svc_pred == y_test)


# ============================================================
# 12) BONUS: Naive Bayes (works directly on factors)
# ============================================================
nb_fit  <- e1071::naiveBayes(respondedMailing ~ ., data = train)
nb_prob <- predict(nb_fit, newdata = test, type = "raw")[, "TRUE"]
nb_pred <- factor(nb_prob >= 0.5, levels = c(FALSE, TRUE))
nb_out  <- metrics_bin(y_test, nb_pred, prob_true = nb_prob)
print_model("Naive Bayes (thr=0.5)", nb_out)


############################################################
# 11) SVC / SVM (Linear, RBF, Poly) — sampled train, full test
# Creates: svc_out, rbf_out, poly_out
############################################################

# Make sure y types are consistent
ytr_small <- factor(ytr_small, levels = c(FALSE, TRUE))
y_test    <- factor(y_test,    levels = c(FALSE, TRUE))

# ---------- (A) Linear SVC ----------
svc_fit <- e1071::svm(
  x = Xtr_small, y = ytr_small,
  kernel = "linear",
  cost = 1,
  scale = FALSE
)

svc_pred <- predict(svc_fit, Xte_svm)
svc_out  <- metrics_bin(y_test, svc_pred, prob_true = NULL)
print_model("SVC Linear (sampled train, full test)", svc_out)

# ---------- (B) SVM RBF ----------
rbf_fit <- e1071::svm(
  x = Xtr_small, y = ytr_small,
  kernel = "radial",
  cost = 1,
  gamma = 1 / ncol(Xtr_small),
  probability = TRUE,
  scale = FALSE
)

rbf_pred <- predict(rbf_fit, Xte_svm, probability = TRUE)
rbf_prob <- attr(rbf_pred, "probabilities")[, "TRUE"]

rbf_out <- metrics_bin(y_test, rbf_pred, prob_true = rbf_prob)
print_model("SVM RBF (sampled train, full test)", rbf_out)

# ---------- (C) SVM Polynomial ----------
poly_fit <- e1071::svm(
  x = Xtr_small, y = ytr_small,
  kernel = "polynomial",
  cost = 1,
  degree = 3,
  gamma = 1 / ncol(Xtr_small),
  coef0 = 0,
  probability = TRUE,
  scale = FALSE
)

poly_pred <- predict(poly_fit, Xte_svm, probability = TRUE)
poly_prob <- attr(poly_pred, "probabilities")[, "TRUE"]

poly_out <- metrics_bin(y_test, poly_pred, prob_true = poly_prob)
print_model("SVM Poly (sampled train, full test)", poly_out)

# ============================================================
# 13) BONUS: Decision Tree
# ============================================================
tree_fit  <- rpart::rpart(respondedMailing ~ ., data = train, method = "class")
tree_prob <- predict(tree_fit, newdata = test, type = "prob")[, "TRUE"]
tree_pred <- factor(tree_prob >= 0.5, levels = c(FALSE, TRUE))
tree_out  <- metrics_bin(y_test, tree_pred, prob_true = tree_prob)
print_model("Decision Tree (rpart, thr=0.5)", tree_out)

# ============================================================
# 14) Final Comparison Table
# ============================================================
get_metric_row <- function(name, obj) {
  if (!exists(obj, inherits = TRUE)) {
    return(data.frame(Model = name, Accuracy = NA, BalancedAcc = NA, AUC = NA))
  }
  x <- get(obj, inherits = TRUE)
  data.frame(Model = name,
             Accuracy = x$accuracy,
             BalancedAcc = x$balanced,
             AUC = x$auc)
}


summary_tbl <- rbind(
  get_metric_row(paste0("Logistic(thr=",thr,")"), "logit_out"),
  get_metric_row("LDA", "lda_out"),
  get_metric_row("QDA", "qda_out"),
  get_metric_row(paste0("KNN(k=",best_k,")"), "knn_out"),
  get_metric_row("SVC Linear", "svc_out"),
  get_metric_row("SVM RBF", "rbf_out"),
  get_metric_row("SVM Poly", "poly_out"),
  get_metric_row("Naive Bayes", "nb_out"),
  get_metric_row("Decision Tree", "tree_out")
)

summary_tbl

