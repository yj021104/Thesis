# 加载必要包
library(randomForest)
library(caret)
library(pdp)
library(ggplot2)
library(haven)
library(dplyr)

# 数据导入
df <- read_sav("D:/yangjing/data/pre.sav")

vars <- c("Investigation", "Gender", "Age", "Marital", "Education", "Family", 
          "Income", "AnnualIncome", "Dailyexpense", "Treatmentexpense", 
          "TimeAllocation", "LifeExperience", "Conflict", 
          "Health", "AIDSHasLittletoDowithOneself", 
          "KeepinTouchwithwhoGetHIV", "AIDSPeoplemesansImmoral", "UsingCondoms", 
          "NMS", "HaventReceivedAnyServices", "ServicesHopedDistributionofPublicityMaterials", 
          "ServicesHopedRegularProvisionofCondoms", "ServicesHopedDoorPublicity", 
          "ServicesHopedHealthDisplayWindows", "ServicesHopedNotNeeded")
target_var <- "AIDSKnowledge"  

# 直接转换所有变量为因子（自动处理标签编码）
df_prepared <- df %>% 
  mutate(
    across(everything(), ~ {
      # 如果变量有标签，使用标签转换
      if (inherits(.x, "labelled")) {
        haven::as_factor(.x)
      } else {
        # 否则直接转换为因子（保持原始数值作为水平）
        as.factor(.x)
      }
    })
  )

# 按插补集分组
imp_data <- split(df_prepared, df_prepared$imp)

# 初始化存储结果
results_list <- list()
importance_list <- list()
oob_list <- list()

# 创建输出目录
dir.create("output_plots", showWarnings = FALSE)

# 遍历每个插补数据集
for (i in seq_along(imp_data)) {
  cat("Processing imputation set", i, "\n")
  
  current_data <- imp_data[[i]]
  
  # 分层抽样
  set.seed(42)
  train_index <- createDataPartition(current_data[[target_var]], p = 0.7, list = FALSE)
  train_data <- current_data[train_index, ]
  test_data <- current_data[-train_index, ]
  
  # 构建随机森林模型
  rf_model <- randomForest(
    x = train_data[, vars],
    y = train_data[[target_var]],
    ntree = 500,
    mtry = floor(sqrt(length(vars))),
    importance = TRUE
  )
  
  # 存储重要性结果
  importance_list[[i]] <- importance(rf_model, type = 2)
  
  # 存储OOB误差
  oob_list[[i]] <- data.frame(
    Tree = 1:rf_model$ntree,
    OOB = rf_model$err.rate[, "OOB"],
    Imputation = paste("Imp", i)
  )
  
  # 获取前5个重要变量
  top5_vars <- names(sort(importance(rf_model, type = 2)[, "MeanDecreaseGini"], decreasing = TRUE))[1:5]
  
  # 创建PDP图
  pdf(file = file.path("output_plots", paste0("PDP_Imp_", i, ".pdf")), 
      width = 15, height = 10)
  par(mfrow = c(2, 3))
  
  for(var in top5_vars) {
    tryCatch({
      pd <- partial(rf_model, 
                    pred.var = var, 
                    train = train_data,
                    which.class = "1",
                    type = "classification")
      
      plot(pd, 
           main = paste("PDP for", var),
           xlab = var,
           ylab = "Predicted Probability")
    }, error = function(e) {
      cat("Error processing", var, ":", conditionMessage(e), "\n")
    })
  }
  dev.off()
  
  # 保存变量重要性图
  pdf(file = file.path("output_plots", paste0("VarImp_Imp_", i, ".pdf")),
      width = 10, height = 8)
  varImpPlot(rf_model, 
             sort = TRUE, 
             n.var = min(20, length(vars)),
             main = paste("Variable Importance - Imputation", i))
  dev.off()
}

# 绘制并保存平均变量重要性图
pdf(file = file.path("output_plots", "Average_Variable_Importance.pdf"),
    width = 10, height = 8)
mean_importance <- Reduce("+", importance_list) / length(importance_list)
par(mar = c(10, 4, 4, 2))
barplot(sort(mean_importance[, "MeanDecreaseGini"], decreasing = TRUE),
        las = 2,
        main = "Average Variable Importance Across Imputations",
        ylab = "Mean Decrease in Gini")
dev.off()

# 绘制并保存OOB误差率图
pdf(file = file.path("output_plots", "OOB_Error_Rates.pdf"),
    width = 10, height = 6)
oob_df <- do.call(rbind, oob_list)
ggplot(oob_df, aes(x = Tree, y = OOB, color = Imputation)) + 
  geom_line() +
  labs(title = "OOB Error Rate Across Trees", 
       x = "Number of Trees", 
       y = "OOB Error Rate") +
  theme_minimal()
dev.off()

# 打印存储位置信息
cat("\nOutput files have been saved to:", normalizePath("output_plots"), "\n")