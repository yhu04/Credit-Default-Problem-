library(randomForest)
library(glmnet)
library(class)
library(tree)
library(pROC)

###################
# Import Data Set #
###################
dataset = read.csv("train.csv",header= TRUE)
head(dataset)
names(dataset)
dataset = dataset[,-1]

##############
# Statistics #
##############
summary(dataset)
# the first problem: missing values. two features have missing values. #

print(table(dataset[,1]))
# the second problem: imbalanced data # 

################################
# Missing Values-train data set#
################################
## Monthly Income ##
# We predict the missing values by considering the relationship between different features. #
missing_value = dataset[,2:10]
missing_vector = is.na(missing_value$MonthlyIncome)
missing_value_output = missing_value[missing_vector,]
missing_value_train = missing_value[!missing_vector,]

missing.fit = lm(MonthlyIncome~.,data=missing_value_train)
missing.predit = predict(missing.fit,newdata=missing_value_output[,-5])
dataset[missing_vector,6] = missing.predit

## Number of Dependents ##
# we calculate the ratio. 0.02616 # 
dataset = dataset[!is.na(dataset$NumberOfDependents),]

######################
# Train and Test Set #
###################### 
train_idx = sample (1:nrow(dataset),0.7*nrow(dataset),replace=FALSE)
train = dataset[train_idx,]
rownames(train) = NULL 
test = dataset[-train_idx,-1]
label = dataset[-train_idx,1]

rownames(test) = NULL 

##################
# under-sampling #
##################
# We use the method of under sampling to solve the problem of imbalanced dataset. # 
number = nrow(train[train[,1]==0,])
under_sample_index = sample(1:number,0.75*number,replace=FALSE)
train_neg = train[train[,1]==0,]
train_pos = train[train[,1]==1,]
train = rbind(train_neg[under_sample_index,],train_pos)
rownames(train) = NULL 

#########
# Model # 
#########

## logistic regression ## 
# regular logistic regression #
train.logic = glm(SeriousDlqin2yrs~., data = train, family = "binomial")
predict.logic = predict(train.logic, newdata=test,type="response")

#Warning message:
#glm.fit: fitted probabilities numerically 0 or 1 occurred 
# This means the data is linearly seperable. That is to say a feature perfectly separate the response into 1's or 0's 
# Then we can use feature selection methods to remove that feature. The method we use here is regulirization. 

# cross validation with lasso #
input = as.matrix(train[,2:11])
output = as.matrix(train[,1])
train.logic.lasso.cv = cv.glmnet(input,output,alpha=1,family='binomial')
predict.logic.lasso.cv = predict(train.logic.lasso.cv, newdata=test,type="response")

## random forest ##
train.randomforest = randomForest(as.factor(SeriousDlqin2yrs)~., data = train, importance=TRUE, ntree=100)
predict.randomforest = predict(train.randomforest, newdata=test,type="response")

## classification tree ##
train.tree = tree(formula=SeriousDlqin2yrs~., data = train)
metric = cv.tree(train.tree)
metric.best = metric$size[which.min(metric$dev)]
train.tree.best= prune.misclass(train.tree , best = metric.best)
predict.tree = predict(train.tree.best, newdata=test,type="response")

## KNN ## 
cl=factor(train[,1])
predict.knn=knn.cv(train[,-1], test[,-1], cl, k = 10, prob=TRUE)

##########################
# Performance Evaluation #
##########################

# We use auc to evaluate the performace. #
auc.logic = auc(predict.logic,label)
auc.logic.lasso.cv = auc(predict.logic.lasso.cv,label)
auc.randomforest = auc(predict.randomforest,label)
auc.tree = auc(predict.tree,label)
auc.knn = auc(predict.knn,label)

AUC = c (auc.logic,auc.logic.lasso.cv,auc.randomforest,auc.tree,auc.knn)
AUC.Value = data.frame(AUC)
names(AUC.Value)=c("logistic regression","lasso logistic regression with cross validation", "random forest", "classification trees" ,"KNN")

# http://www.r-bloggers.com/classification-trees/  
# https://citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics/ # 
