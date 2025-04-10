library("randomForest")

#数据导入
df <- read_sav("D:/yangjing/data/pre.sav")

# 定义变量
vars <- c("Investigation", "Gender", "Age", 
          "Marital.2", "Marital.3", "Marital.4", 
          "Education.2", "Education.3", "Education.4", "Education.5", 
          "Family.2", "Family.3", "Family.4", "Family.5", 
          "Income.2", "Income.3", "Income.4", "Income.5", 
          "AnnualIncome.2", "AnnualIncome.3", "AnnualIncome.4", 
          "Dailyexpense", "Treatmentexpense", 
          "TimeAllocation.2", "TimeAllocation.3", 
          "TimeAllocation.4", "TimeAllocation.5", "TimeAllocation.6", 
          "LifeExperience.2", "LifeExperience.3", 
          "LifeExperience.4", "LifeExperience.5", 
          "Conflict.2", "Conflict.3", "Conflict.4", 
          "Health.2", "Health.3", "Health.4", "Health.5", 
          "AIDSHasLittletoDowithOneself.2", "AIDSHasLittletoDowithOneself.3",  
          "KeepinTouchwithwhoGetHIV.2", "KeepinTouchwithwhoGetHIV.3", 
          "AIDSPeoplemesansImmoral.2", "AIDSPeoplemesansImmoral.3",  
          "UsingCondoms.2", "UsingCondoms.3", 
          "NMS", "HaventReceivedAnyServices", 
          "ServicesHopedDistributionofPublicityMaterials", 
          "ServicesHopedRegularProvisionofCondoms", "ServicesHopedDoorPublicity", 
          "ServicesHopedHealthDisplayWindows", "ServicesHopedNotNeeded")
target_var <- "AIDSKnowledge"  

#因变量确定转化为因子型
newdata$class <- as.factor(newdata$class)  

# 按插补集分组
imp_data <- split(df, df$imp)

# 初始化存储每个插补集结果的列表
results_list <- vector("list", length(imp_data))

# 遍历每个插补数据集，构建随机森林
for (i in seq_along(imp_data)) {
  # 提取当前插补数据集
  data <- imp_data[[i]]
  
  # 分层交叉验证（保证每个折的比例与原数据一致），k取10
  folds <- createFolds(factor(imp_data$AIDSKnowledge), k = 10, list = FALSE)
  
  # 寻找最优参数mtry，即指定节点中用于二叉树的最佳变量个数
  n<-length(names(train_data))     #计算数据集中自变量个数，等同n=ncol(train_data)
  rate=1     #设置模型误判率向量初始值
  
  for(i in 1:(n-1)){
    set.seed(1234)
    rf_train<-randomForest(as.factor(train_data$IS_LIUSHI)~.,data=train_data,mtry=i,ntree=1000)
    rate[i]<-mean(rf_train$err.rate)   #计算基于OOB数据的模型误判率均值
    print(rf_train)    
  }
  
  rate     #展示所有模型误判率的均值
  plot(rate)
  min(rate)     #找出最小的模型误判率均值
  which.min(rate)     #找出最小的模型误判率均值对应的mtry值
  
  #寻找最佳参数ntree，即指定随机森林所包含的最佳决策树数目
  set.seed(42)
  rf_train<-randomForest(as.factor(train_data$IS_LIUSHI)~.,data=train_data,mtry=12,ntree=1000)
  plot(rf_train)    #绘制模型误差与决策树数量关系图  
  legend(800,0.02,"IS_LIUSHI=0",cex=0.9,bty="n")    
  legend(800,0.0245,"total",cex=0.09,bty="n")    
  legend(800,0.027,"IS_LIUSHI=1",cex=0.9,bty="n")
  
  #构建随机森林模型
  set.seed(42)
  rf_train<-randomForest(as.factor(train_data$IS_LIUSHI)~.,data=train_data,mtry=12,ntree=400,importance=TRUE,proximity=TRUE)    
  
  #模型评估
  rf_train
  plot(rf_train)    #绘制模型误差与决策树数量关系图
  legend(800,0.02,"IS_LIUSHI=0",cex=0.9,bty="n")
  legend(800,0.0245,"total",cex=0.09,bty="n")
  legend(800,0.027,"IS_LIUSHI=1",cex=0.9,bty="n")
  
  #输出变量重要性:分别从精确度递减和均方误差递减的角度来衡量重要程度
  importance<-importance(rf_train) 
  write.csv(importance,file="E:/模型搭建/importance.csv",row.names=T,quote=F)
  barplot(rf_train$importance[,1],main="输入变量重要性测度指标柱形图")
  box()
  
  #提取随机森林模型中以准确率递减方法得到维度重要性值。type=2为基尼系数方法
  importance(rf_train,type=2)
  varImpPlot(x=rf_train,sort=TRUE,n.var=nrow(rf_train$importance),main="输入变量重要性测度散点图")
  
  #展示训练集训练的随机森林模型信息
  print(rf_train)    #展示随机森林模型简要信息
  hist(treesize(rf_train))   #展示随机森林模型中每棵决策树的节点数
  max(treesize(rf_train));min(treesize(rf_train))
  #展示数据集在二维情况下各类别的具体分布情况
  MDSplot(rf_train,train_data$IS_OFF_USER,palette=rep(1,2),pch=as.numeric(train_data$IS_LIUSHI))    
  
  #测试集进行检测
  pred<-predict(rf_train,newdata=test_data)  
  pred_out_1<-predict(object=rf_train,newdata=test_data,type="prob")  #输出概率
  table <- table(pred,test_data$IS_LIUSHI)  
  sum(diag(table))/sum(table)  #预测准确率
  plot(margin(rf_train,test_data$IS_LIUSHI),main=观测值被判断正确的概率图)
  
  # 计算各插补集测试数据准确率、精确率、F1 评分、AUC
  accuracy_list[[i]] <- confusionMatrix(pred, truth)$overall['Accuracy']
  auc_list[[i]] <- roc(truth, pred)$auc
  
  # 可视化测试集的分类结果
  plot(roc(truth, pred), main = "ROC Curve", col = "blue", lwd = 2)
  
}

#各插补集准确率、精确率、F1 评分、AUC汇总（平均值与标准差）
accuracy_list <- unlist(accuracy_list)
auc_list <- unlist(auc_list)
mean(accuracy_list)
sd(accuracy_list)
mean(auc_list)
sd(auc_list)

#可视化平均ROC曲线
truth <- unlist(lapply(imp_data, function(x) x$diagnosis))
pred <- unlist(lapply(imp_data, function(x) predict(x, newdata = x)))
roc_curve <- roc(truth, pred)
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)

#保存ROC曲线
pdf("D:/yangjing/data/ROC.pdf")
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)
dev.off()
