# Load necessary libraries
library(EBImage)
library(caret)
library(stats)
library(ggplot2)

# Function to generate a Gaussian kernel
generate_gaussian_kernel <- function(size, sigma) {
  x <- seq(-floor(size / 2), floor(size / 2), length.out = size)
  kernel <- outer(x, x, function(x, y) exp(-(x^2 + y^2) / (2 * sigma^2)))
  kernel <- kernel / sum(kernel)  # Normalize
  return(kernel)
}

# Function to generate a motion blur kernel
generate_motion_kernel <- function(size, angle) {
  if (size %% 2 == 0) stop("Size must be odd to ensure a central element.")
  kernel <- matrix(0, nrow = size, ncol = size)
  center <- ceiling(size / 2)
  radian <- angle * pi / 180
  length <- floor(size / 2)
  for (i in -length:length) {
    x <- round(center + i * cos(radian))
    y <- round(center + i * sin(radian))
    if (x >= 1 && x <= size && y >= 1 && y <= size) kernel[y, x] <- 1
  }
  kernel <- kernel / sum(kernel)
  return(kernel)
}

# Function to apply a blur kernel to an image
apply_blur <- function(image_path, kernel) {
  image <- readImage(image_path)
  image <- channel(image, "gray")  # Convert to grayscale
  blurred <- filter2(image, kernel)
  return(blurred)
}

# Function to apply FFT shift
fftshift <- function(image) {
  nr <- nrow(image)
  nc <- ncol(image)
  image_shifted <- image[c((nr %/% 2 + 1):nr, 1:(nr %/% 2)), c((nc %/% 2 + 1):nc, 1:(nc %/% 2))]
  return(image_shifted)
}

# Function to apply FFT to an image
apply_fft <- function(image) {
  fft_image <- fft(image)
  shifted_fft <- abs(fftshift(Mod(fft_image)))
  magnitude_spectrum <- log(shifted_fft + 1)
  return(magnitude_spectrum)
}

# Function to extract features from an image
extract_features <- function(image) {
  sobel_x <- filter2(image, matrix(c(
    -1, -2,  0,  2,  1,
    -4, -8,  0,  8,  4,
    -6, -12, 0, 12, 6,
    -4, -8,  0,  8,  4,
    -1, -2,  0,  2,  1
  ), nrow = 3))
  sobel_y <- filter2(image, matrix(c(
    -1, -4, -6, -4, -1,
    -2, -8, -12, -8, -2,
    0,  0,   0,  0,  0,
    2,  8,  12,  8,  2,
    1,  4,   6,  4,  1
  ), nrow = 3))
  sobel_combined <- sqrt(sobel_x^2 + sobel_y^2)
  spatial_features <- mean(sobel_combined)
  magnitude_spectrum <- apply_fft(image)
  frequency_features <- mean(magnitude_spectrum)
  return(c(spatial_features, frequency_features))
}

# Define blur kernels
blur_kernels <- list(
  generate_gaussian_kernel(21, 1), 
  generate_gaussian_kernel(21, 2),
  generate_gaussian_kernel(21, 3), 
  generate_gaussian_kernel(21, 4),
  generate_gaussian_kernel(21, 5), 
  generate_gaussian_kernel(21, 6),
  generate_gaussian_kernel(21, 7),
  generate_motion_kernel(21, 0), 
  generate_motion_kernel(21, 45), 
  generate_motion_kernel(21, 90)
)

# Relative folder containing images
image_folder <- "./DIV2K_valid_HR"
image_files <- list.files(image_folder, pattern = "\\.(jpg|png)$", full.names = TRUE)

# Prepare the data
data <- list()
labels <- list()

for (i in seq_along(image_files)) {
  cat(sprintf("Processing image %d/%d: %s\n", i, length(image_files), basename(image_files[i])))
  tryCatch({
    kernel <- sample(blur_kernels, 1)[[1]]
    blurred <- apply_blur(image_files[i], kernel)
    features <- extract_features(blurred)
    data[[i]] <- features
    labels[[i]] <- as.vector(kernel)
  }, error = function(e) {
    cat(sprintf("Error processing file %s: %s\n", image_files[i], e$message))
  })
}

# Convert data into matrices
X <- do.call(rbind, data)
y <- do.call(rbind, labels)

# Normalize data
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
X <- apply(X, 2, normalize)
y <- normalize(y)

# Split into train and test sets
set.seed(42)
train_indices <- sample(1:nrow(X), size = 0.8 * nrow(X), replace = FALSE)

X_train <- X[train_indices, ]
y_train <- y[train_indices, ]
X_test <- X[-train_indices, ]
y_test <- y[-train_indices, ]

# Check dimensions
cat("Dimensions of X_train :", dim(X_train), "\n")
cat("Dimensions of y_train :", dim(y_train), "\n")

# Convert to data frames
X_train_df <- as.data.frame(X_train)
colnames(X_train_df) <- paste0("X", seq_len(ncol(X_train)))

X_test_df <- as.data.frame(X_test)
colnames(X_test_df) <- paste0("X", seq_len(ncol(X_test)))

# Generalized function to train multi-output models
train_multi_output_model <- function(X_train, y_train, method, tuneGrid, trControl, model_dir) {
  models <- list()
  if (!dir.exists(model_dir)) dir.create(model_dir, recursive = TRUE)
  
  for (i in 1:ncol(y_train)) {
    cat("Training", method, "model for output", i, "/", ncol(y_train), "\n")
    models[[i]] <- train(
      X_train, y_train[, i],
      method = method,
      tuneGrid = tuneGrid,
      trControl = trControl
    )
    saveRDS(models[[i]], file = file.path(model_dir, paste0("model_", method, "_output_", i, ".rds")))
  }
  cat("All", method, "models have been saved in folder:", model_dir, "\n")
  return(models)
}

# Function to predict multi-output models
predict_multi_output_model <- function(models, X_test) {
  predictions <- matrix(NA, nrow = nrow(X_test), ncol = length(models))
  for (i in 1:length(models)) {
    predictions[, i] <- predict(models[[i]], X_test)
  }
  return(predictions)
}

# Define train control
train_control <- trainControl(method = "cv", number = 5)

# Train and save models for RF, kNN, and SVM
results <- list()

# 1. Random Forest
rf_tuneGrid <- expand.grid(mtry = 2:5)
results$RF <- list()
results$RF$models <- train_multi_output_model(
  X_train = X_train_df,
  y_train = y_train,
  method = "rf",
  tuneGrid = rf_tuneGrid,
  trControl = train_control,
  model_dir = "./models_rf"
)
results$RF$predictions <- predict_multi_output_model(results$RF$models, X_test_df)

# 2. k-Nearest Neighbors
knn_tuneGrid <- expand.grid(k = seq(3, 15, 2))
results$KNN <- list()
results$KNN$models <- train_multi_output_model(
  X_train = X_train_df,
  y_train = y_train,
  method = "knn",
  tuneGrid = knn_tuneGrid,
  trControl = train_control,
  model_dir = "./models_knn"
)
results$KNN$predictions <- predict_multi_output_model(results$KNN$models, X_test_df)

# 3. Support Vector Machines
svm_tuneGrid <- expand.grid(
  sigma = c(0.01, 0.1, 1),
  C = c(1, 10, 100)
)
results$SVM <- list()
results$SVM$models <- train_multi_output_model(
  X_train = X_train_df,
  y_train = y_train,
  method = "svmRadial",
  tuneGrid = svm_tuneGrid,
  trControl = train_control,
  model_dir = "./models_svm"
)
results$SVM$predictions <- predict_multi_output_model(results$SVM$models, X_test_df)

# 4. Gradient Boost Machine (GBM)
gbm_tuneGrid <- expand.grid(
  n.trees = c(50, 100),
  interaction.depth = c(1, 3),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = 10
)
results$GBM <- list()
results$GBM$models <- train_multi_output_model(
  X_train = X_train_df,
  y_train = y_train,
  method = "gbm",
  tuneGrid = gbm_tuneGrid,
  trControl = train_control,
  model_dir = "./models_gbm"
)
results$GBM$predictions <- predict_multi_output_model(results$GBM$models, X_test_df)


# Evaluate metrics for all models
calculate_metrics <- function(actual, predicted) {
  mae_per_column <- colMeans(abs(actual - predicted))
  mse_per_column <- colMeans((actual - predicted)^2)
  ss_total <- colSums((actual - colMeans(actual))^2)
  ss_residual <- colSums((actual - predicted)^2)
  r2_per_column <- 1 - (ss_residual / ss_total)
  mae_global <- mean(mae_per_column)
  mse_global <- mean(mse_per_column)
  r2_global <- mean(r2_per_column)
  return(list(mae_per_column, mse_per_column, r2_per_column, mae_global, mse_global, r2_global))
}

# Calculate metrics for each model
metrics <- list()
metrics_df <- data.frame(
  Model = character(0),
  MAE_Global = numeric(0),
  MSE_Global = numeric(0),
  R2_Global = numeric(0),
  stringsAsFactors = FALSE
)

for (model_name in names(results)) {
  cat("Evaluating model:", model_name, "\n")
  metrics[[model_name]] <- calculate_metrics(y_test, results[[model_name]]$predictions)
}

# Display global results
for (model_name in names(metrics)) {
  cat("\nModel:", model_name, "\n")
  cat("Global MAE:", metrics[[model_name]][[4]], "\n")
  cat("Global MSE:", metrics[[model_name]][[5]], "\n")
  cat("Global R²:", metrics[[model_name]][[6]], "\n")
  
  metrics_df <- rbind(metrics_df, data.frame(
    Model = model_name,
    MAE_Global = metrics[[model_name]][[4]],
    MSE_Global = metrics[[model_name]][[5]],
    R2_Global = metrics[[model_name]][[6]]
  ))
}


#save the metrics results into a file
write.csv(metrics_df, file = "model_metrics.csv", row.names = FALSE)

cat("Metrics have been saved to 'model_metrics.csv'.\n")



