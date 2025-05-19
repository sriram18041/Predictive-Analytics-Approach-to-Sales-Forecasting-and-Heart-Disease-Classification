# install.packages(c("TTR", "tseries", "urca"))

# Load required libraries
library(TTR)      # For EMA smoothing
library(tseries)  # For ARMA and ADF tests
library(urca)     # For cointegration and VECM

# 1. Generating Synthetic Time Series Dataset
set.seed(456)
time <- seq(as.Date("2015-01-01"), by = "month", length.out = 120)  # 120 months
TV_Advertising <- 50 + 0.3 * 1:120 + 10 * sin(2 * pi * (1:120) / 12) + rnorm(120, 0, 5)
Sales <- 100 + 0.6 * TV_Advertising + 15 * sin(2 * pi * (1:120) / 12) + rnorm(120, 0, 10)
data <- data.frame(Date = time, Sales = Sales, TV_Advertising = TV_Advertising)

# 2. Smoothing (EMA)
data$Sales_EMA <- EMA(data$Sales, n = 12)  # 12-month EMA
# Plot original vs smoothed Sales
plot(data$Date, data$Sales, type = "l", col = "blue", ylab = "Sales", xlab = "Date", 
     main = "Sales with EMA Smoothing")
lines(data$Date, data$Sales_EMA, col = "red")
legend("topleft", legend = c("Original Sales", "EMA Smoothed"), col = c("blue", "red"), lty = 1)

# 3. ARMA Analysis
Sales_diff <- diff(data$Sales)
# Plot ACF and PACF
par(mfrow = c(1, 2))  # Side-by-side plots
acf(Sales_diff, main = "ACF of Differenced Sales")
pacf(Sales_diff, main = "PACF of Differenced Sales")
# Fit ARMA(1,1) model (adjust order based on ACF/PACF if needed)
arma_model <- arima(Sales_diff, order = c(1, 0, 1))
# Check residuals for white noise
residuals <- residuals(arma_model)
acf(residuals, main = "ACF of ARMA Residuals")
box_test <- Box.test(residuals, type = "Ljung-Box")
print("Ljung-Box Test for Residuals:")
print(box_test)

# 4. Stationary Testing (ADF Test)
cat("\nADF Test for Sales:\n")
print(adf.test(data$Sales, alternative = "stationary"))
cat("\nADF Test for TV_Advertising:\n")
print(adf.test(data$TV_Advertising, alternative = "stationary"))
# Difference and re-test
Sales_diff <- diff(data$Sales)
TV_Advertising_diff <- diff(data$TV_Advertising)
cat("\nADF Test for Differenced Sales:\n")
print(adf.test(Sales_diff, alternative = "stationary"))
cat("\nADF Test for Differenced TV_Advertising:\n")
print(adf.test(TV_Advertising_diff, alternative = "stationary"))

# 5. Multivariate Regression Model
cointegration_test <- ca.jo(data[, c("Sales", "TV_Advertising")], type = "trace", 
                            ecdet = "trend", K = 2)
cat("\nCointegration Test (Trace):\n")
print(summary(cointegration_test))
# Fit VECM
vecm_model <- cajorls(cointegration_test, r = 1)
cat("\nVECM Model Summary:\n")
print(summary(vecm_model))

# Reset plotting parameters
par(mfrow = c(1, 1))

# End of script