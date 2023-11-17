
if(!require(caret)) install.packages("caret") 
if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(neuralnet)) install.packages("neuralnet") 
if(!require(rpart)) install.packages("rpart") 
if(!require(rpart.plot)) install.packages("rpart.plot") 
if(!require(scales)) install.packages("scales") 


library(caret)
library(tidyverse)
library(neuralnet)
library(rpart)
library(rpart.plot)
library(scales)

#show the data

data("iris")

colnames(iris)<-c("SL","SW","PL","PW","Species")

rbind(
  head(iris,n=2),
  
  head(iris[which(iris$Species=="versicolor"),],n=2),
  
  head(iris[which(iris$Species=="virginica"),],n=2)
)


#mean and range
iris_1<-iris %>% 
  group_by(Species)  %>% summarize(AVG_SL=mean(SL),MIN_SL=min(SL),MAX_SL=max(SL),
                                   AVG_SW=mean(SW),MIN_SW=min(SW),MAX_SW=max(SW),
                                   AVG_PL=mean(PL),MIN_PL=min(PL),MAX_PL=max(PL),
                                   AVG_PW=mean(PW),MIN_PW=min(PW),MAX_PW=max(PW)
  )

head(iris_1)

#plot all potential predictors
pairs(iris[, 1:4], col = factor(iris$Species), pch = 19, oma=c(2,5,2,2), data = iris)
par(xpd = TRUE)
legend("topleft", fill = unique(iris$Species),cex=0.5, legend = c( levels(iris$Species)))

#choose the right predictors

iris<-iris %>%  select("PL","PW","Species")


#scaling the data

maxval<-apply(iris[,1:2],2, max)
minval<-apply(iris[,1:2],2, min)

iris_scale<-as.data.frame(scale(iris[,1:2],center=minval,scale=(maxval-minval)))
iris_data_round<-cbind(round(iris_scale,4),Species=iris$Species)
iris_data<-cbind(iris_scale,Species=iris$Species)

head(iris_data_round)

rbind(
  head(iris_data_round,n=2),
  
  head(iris_data_round[which(iris_data_round$Species=="versicolor"),],n=2),
  
  head(iris_data_round[which(iris_data_round$Species=="virginica"),],n=2)
)

pairs(iris_data[, 1:2], col = factor(iris_data$Species), pch = 19, oma=c(2,5,2,2), data = iris_data)
par(xpd = TRUE)
legend("topleft", fill = unique(iris_data$Species),cex=0.5, legend = c( levels(iris_data$Species)))


#Partitioning the data to train and test

set.seed(1, sample.kind="Rounding") # if using R 3.6 or later

index_train <- createDataPartition(iris_data$Species, p=0.7, list = FALSE)


test <-  iris_data[-index_train,]  

train <- iris_data[index_train,]

length(test$Species)
table(train$Species)



# classification Decision Trees and plot it

train_rpart <- rpart(Species~., data = train)


rpart.plot(train_rpart)


# compute accuracy in train

Predict_rpart_train <-  predict(train_rpart, newdata = train, type = "class")

table(train$Species, Predict_rpart_train)

accuracy_rpart_train<- percent(mean(train$Species==Predict_rpart_train), accuracy = 0.1) 
error_rate_rpart_train<-percent(mean(train$Species!=Predict_rpart_train), accuracy = 0.1) 
accuracy_rpart_train
error_rate_rpart_train

# compute accuracy in test

Predict_rpart_tests <-  predict(train_rpart, newdata = test, type = "class")

table(test$Species, Predict_rpart_tests)

accuracy_rpart_test<- percent(mean(test$Species==Predict_rpart_tests), accuracy = 0.1) 
error_rate_rpart_test<-percent(mean(test$Species!=Predict_rpart_tests), accuracy = 0.1) 

accuracy_rpart_test
error_rate_rpart_test

# classification on training data with randomForest

Predict_randomF_train = randomForest::randomForest(Species ~ ., data = train)
Predict_randomF_train

# classification on test data with randomForest

Predict_randomF_tests = randomForest::randomForest(Species ~ ., data = test)
Predict_randomF_tests

#creating a neural net work on train_set to predict the Species
rm(ANN)
ANN<-neuralnet:: neuralnet(Species  ~ ., train , hidden =c(4,2))

plot(ANN,rep = "best")

# compute accuracy in train

pred_set = neuralnet:: compute(ANN,train[,c(1,2,3)])

pred_df <- data_frame()

n=c(length(train$Species))

for (i in 1:n) {
  pred_df<- rbind(pred_df, which.max(pred_set$net.result[i,]))
}

pred_df$X1L<- gsub(1,"setosa", pred_df$X1L)
pred_df$X1L<- gsub(2,"versicolor", pred_df$X1L)
pred_df$X1L<- gsub(3,"virginica", pred_df$X1L)

prediction <- pred_df$X1L 
reference <- train$Species


accuracy_nna_train<-mean(prediction == reference)
error_nna_train<-mean(prediction != reference)

table(prediction, reference)

# compute accuracy in test

perd_test<- predict(ANN,test,type="class")
reference_test <-test$Species

validation_df <-data.frame(x=apply(perd_test,1,which.max))

validation_df<-ifelse( validation_df$x==1,"setosa",
                       ifelse(validation_df$x==2,"versicolor",
                              "virginica"))

table(reference_test,validation_df)

accuracy_nna_test<-mean(reference_test==validation_df) 
error_nna_test<-mean(reference_test!=validation_df) 
#check the the best method 

N_seeds<- c(1,12,123,1234,12345,123456,2089, 2676, 4831, 4261,132,2344,2334,3366,5465,
            380,99,876,594,1998,35,45,67,878,23,8766,3401,31,84,96)
length(N_seeds)
rm(models_results_final,models_results)

models_results <- data_frame()
models_results_final <- data_frame()

for (i in 1:length(N_seeds)) {
  set.seed(N_seeds[i]) # if using R 3.6 or later
  #create data
  index_train <- createDataPartition(iris_data$Species, p=0.7, list = FALSE)
  test <-  iris_data[-index_train,]  
  train <- iris_data[index_train,]
  train_rpart <- rpart(Species~., data = train)
  #rpart
  Predict_rpart_tests <-  predict(train_rpart, newdata = test, type = "class")
  accuracy_rpart_test<- mean(test$Species==Predict_rpart_tests)
  error_rate_rpart_test<-mean(test$Species!=Predict_rpart_tests) 
  #NNA
  ANN<-neuralnet:: neuralnet(Species  ~ ., train , hidden =c(4,2))
  perd_test<- predict(ANN,test,type="class")
  reference_test <-test$Species
  validation_df <-data.frame(x=apply(perd_test,1,which.max))
  validation_df<-ifelse( validation_df$x==1,"setosa",
                         ifelse(validation_df$x==2,"versicolor",
                                "virginica"))
  table(reference_test,validation_df)
  accuracy_ann_test<-mean(reference_test==validation_df) 
  error_ann_test<-mean(reference_test!=validation_df)
  #summery
  models_results <- bind_rows(tibble(N_seeds ,accuracy_rpart=accuracy_rpart_test,error_rate_rpart=error_rate_rpart_test,
                                     accuracy_ann=accuracy_ann_test,error_ann=error_ann_test,
                                     diff_accuracy=accuracy_ann-accuracy_rpart
  ))
  models_results_final<-rbind(models_results_final,models_results[i,])
}

models_results_final<-models_results_final %>% mutate(
  accuracy_rpart= percent(accuracy_rpart, accuracy = 0.1),
  error_rate_rpart= percent(error_rate_rpart , accuracy = 0.1),
  accuracy_ann= percent(accuracy_ann , accuracy = 0.1),
  error_ann= percent(error_ann , accuracy = 0.1),
  diff_accuracy= percent(diff_accuracy, accuracy = 0.1),
)

summary_count<-models_results_final%>%summarise(count_0=length(which(diff_accuracy==0)),
                                                count_p=length(which(diff_accuracy>0)),
                                                count_n=length(which(diff_accuracy<0))
)

models_results_final<-models_results_final %>% mutate(
  accuracy_rpart= percent(accuracy_rpart, accuracy = 0.1),
  error_rate_rpart= percent(error_rate_rpart , accuracy = 0.1),
  accuracy_ann= percent(accuracy_ann , accuracy = 0.1),
  error_ann= percent(error_ann , accuracy = 0.1),
  diff_accuracy= percent(diff_accuracy, accuracy = 0.1),
)

head(models_results_final)

summary_count
