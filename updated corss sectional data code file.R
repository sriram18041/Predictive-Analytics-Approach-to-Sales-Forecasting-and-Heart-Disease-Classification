# Group members: Srirama Krishna Sarma Anappindi, Venkata Somanath Chittilla, Aditya Ashok Giakwad, Akhil Rajkumar

# Check if the file exists
if (!file.exists("heart.csv")) {
  stop("File 'heart.csv' not found. Check the filename or path with dir().")
}

# Load required libraries
if (!require(cluster)) {
  install.packages("cluster")
  library(cluster)
}
if (!require(randomForest)) {
  install.packages("randomForest")
  library(randomForest)
}
library(stats)  # For kmeans() and glm()

# Load the dataset
data <- read.csv("heart.csv")
cat("Dataset loaded. Number of rows:", nrow(data), "Number of columns:", ncol(data), "\n")
print("First few rows of the dataset:")
print(head(data))

# Prepare data for classification: Convert 'num' to binary (0 = no disease, 1 = disease)
data$target <- ifelse(data$num > 0, 1, 0)  # Adjust if your column is already 'target'
cat("Target variable created (binary: 0 = no disease, 1 = disease). Distribution:\n")
print(table(data$target))

# Define all variables used in classification
class_vars <- c("target", "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal")

# Check for NAs in classification variables and remove rows with any NAs
na_counts_class <- colSums(is.na(data[, class_vars]))
cat("NA counts in classification variables:\n")
print(na_counts_class)
if (any(na_counts_class > 0)) {
  cat("Removing rows with NA values in classification variables...\n")
  data_clean <- data[complete.cases(data[, class_vars]), ]
  cat("Rows after removing NAs for classification:", nrow(data_clean), "\n")
} else {
  data_clean <- data
}

# --- Classification ---
# Split data into training and testing sets
set.seed(123)  # For reproducibility
train_idx <- sample(1:nrow(data_clean), 0.7 * nrow(data_clean))  # 70% for training
train <- data_clean[train_idx, ]
test <- data_clean[-train_idx, ]

# Ensure categorical variables are factors and align levels
categorical_vars <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "thal")
for (var in categorical_vars) {
  if (var %in% colnames(data_clean)) {
    train[[var]] <- as.factor(train[[var]])
    test[[var]] <- factor(test[[var]], levels = levels(train[[var]]))
    test <- test[!is.na(test[[var]]), ]  # Remove test rows with new levels
  }
}
cat("Categorical variables aligned. Test rows after filtering new levels:", nrow(test), "\n")

# Logistic Regression
logit_model <- glm(target ~ age + sex + cp + trestbps + chol + fbs + restecg + thalch + exang + oldpeak + slope + ca + thal, 
                   data = train, family = "binomial")
logit_pred <- predict(logit_model, test, type = "response")
logit_pred_class <- ifelse(logit_pred > 0.5, 1, 0)
logit_accuracy <- mean(logit_pred_class == test$target, na.rm = TRUE)
cat("Logistic Regression Accuracy:", logit_accuracy, "\n")

# Random Forest
rf_model <- randomForest(as.factor(target) ~ age + sex + cp + trestbps + chol + fbs + restecg + thalch + exang + oldpeak + slope + ca + thal, 
                         data = train, ntree = 100)
rf_pred <- predict(rf_model, test)
rf_accuracy <- mean(rf_pred == test$target, na.rm = TRUE)
cat("Random Forest Accuracy:", rf_accuracy, "\n")

# Define quantitative variables for clustering
quant_vars <- c("age", "trestbps", "chol", "thalch", "oldpeak", "ca")

# Check if all variables exist
missing_vars <- quant_vars[!quant_vars %in% colnames(data_clean)]
if (length(missing_vars) > 0) {
  stop(paste("Missing variables:", paste(missing_vars, collapse = ", ")))
}

# Diagnose NA values in quant_vars (should be none after cleaning)
na_counts <- colSums(is.na(data_clean[, quant_vars]))
cat("NA counts in quantitative variables (post-cleaning):\n")
print(na_counts)

# Check data types
data_types <- sapply(data_clean[, quant_vars], class)
cat("Data types of quantitative variables:\n")
print(data_types)

# Handle non-numeric columns (shouldn’t be needed after initial cleaning, but kept for safety)
for (var in quant_vars) {
  if (!is.numeric(data_clean[[var]])) {
    cat("Converting", var, "to numeric...\n")
    data_clean[[var]] <- as.numeric(as.character(data_clean[[var]]))
    if (any(is.na(data_clean[[var]]))) {
      cat("Warning: Conversion of", var, "introduced NAs.\n")
    }
  }
}

# Standardize the data
scaled_data <- scale(data_clean[, quant_vars])
if (is.null(scaled_data) || any(is.na(scaled_data))) {
  stop("Failed to create scaled_data after cleaning. Check data integrity.")
}
cat("Data standardized successfully. Preview of scaled_data:\n")
print(head(scaled_data))

# K-Means Clustering
set.seed(123)
kmeans_result <- kmeans(scaled_data, centers = 3, nstart = 25)
cat("K-Means clustering completed. Cluster assignments for the first few observations:\n")
print(head(kmeans_result$cluster))
cat("Cluster sizes:\n")
print(table(kmeans_result$cluster))
cat("Within-cluster sum of squares by cluster:\n")
print(kmeans_result$withinss)
cat("Total within-cluster sum of squares:", kmeans_result$tot.withinss, "\n")

# K-Means Plot
dev.new()  # Open new graphic window
if (exists("scaled_data") && exists("kmeans_result")) {
  plot(scaled_data[, "age"], scaled_data[, "chol"], col = kmeans_result$cluster, 
       pch = 19, main = "K-Means Clusters (Age vs Cholesterol)", 
       xlab = "Age (scaled)", ylab = "Cholesterol (scaled)")
  legend("topright", legend = 1:3, col = 1:3, pch = 19, title = "Cluster")
  cat("K-Means scatterplot displayed in a new window.\n")
} else {
  stop("Cannot plot K-Means: scaled_data or kmeans_result is missing.")
}

# Calculate distance matrix (for hierarchical clustering)
dist_matrix <- dist(scaled_data)
if (is.null(dist_matrix)) {
  stop("Failed to create dist_matrix. Check scaled_data.")
}
cat("Distance matrix created successfully with", length(dist_matrix), "elements.\n")

# Perform hierarchical clustering
hclust_result <- hclust(dist_matrix, method = "ward.D2")
if (is.null(hclust_result)) {
  stop("Failed to create hclust_result. Check dist_matrix.")
}
cat("Hierarchical clustering completed successfully.\n")

# Cut the dendrogram into 3 clusters
clusters_h <- cutree(hclust_result, k = 3)
cat("Hierarchical cluster assignments for the first few observations:\n")
print(head(clusters_h))

# Plot the dendrogram
dev.new()  # Open a new graphic window
plot(hclust_result, main = "Hierarchical Clustering Dendrogram", xlab = "Patients", ylab = "Height", hang = -1)
rect.hclust(hclust_result, k = 3, border = "orange")
cat("Dendrogram displayed in a new window.\n")