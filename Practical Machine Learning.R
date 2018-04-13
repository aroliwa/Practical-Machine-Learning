library(Amelia)
library(caret)
library(randomForest)
library(dplyr)
library(doSNOW)

training <- read.csv(file = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                     header = TRUE, na.strings=c("", "NA"))
testing <- read.csv(file = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                    header = TRUE, na.strings=c("", "NA"))

missmap(obj = training, y.cex=0.5, x.cex=0.7)

cols <- sapply(X = training, FUN = function(X) sum(is.na(X)) == 0)
training <- training[, cols]
rm(cols)

glimpse(training[,1:10])
training <- training[,-c(1:7)]


cl <- makeCluster(2, type="SOCK")
registerDoSNOW(cl)

set.seed(123)

train_control <- trainControl(method = "cv", number = 3, allowParallel = TRUE, verboseIter = TRUE)

model.rf <- train(classe ~ ., data = training,
                  method = "rf",
                  importance = TRUE,
                  trControl = train_control)

stopCluster(cl)


plot(model.rf$finalModel, main="Random Forest (Error Rate vs. Number of Trees)")

model.rf

varImpPlot(model.rf$finalModel)

predict(object = model.rf, newdata = testing)
