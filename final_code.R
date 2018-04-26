install.packages(c("ROCR", "caTools", "MASS"))
library(dplyr)
library(ggplot2)
library(GGally)
library(car) # for VIF
library(caTools)
library(ROCR)
library(MASS)

# load data
moviee <- read.csv("data_ready.csv", stringsAsFactors=FALSE,header=TRUE,sep=";", na.strings="NA")
movie <- read.csv("movie_metadata_modified_ohe.csv")
set.seed(144)
# remove null values
movie_new <- na.omit(movie)
moviee = moviee[!is.na(moviee$Gross_final),]

moviee$Successful = NA
moviee$Successful = as.factor(as.numeric(moviee$Gross_final >= 2*moviee$Bugdet_final))
for(i in 1:ncol(moviee)){
  moviee[is.na(moviee[,i]), i] <- mean(moviee[,i], na.rm = TRUE)
}
str(moviee)
# split into training and testing sets
movie.train <- filter(moviee,Year>=2002)
movie.test <- filter(moviee,Year<2002)


split=sample(seq_len(nrow(moviee)),size=floor(0.7*nrow(moviee)))
movie.train=moviee[split,]
movie.test=moviee[-split,]


#--------------Boosting------------------------#
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(gbm)
library(caTools)
library(dplyr)
library(ggplot2)

OSR2 <- function(predictions, test, train) {
  SSE <- sum((test - predictions)^2)
  SST <- sum((test - mean(train))^2)
  r2 <- 1 - SSE/SST
  return(r2)
}
 

mod.boost <- gbm(log(Gross_final) ~   Year + label_country + label_language + isAction
                 + isAdventure +	isAnimation	
                 + isBiography	
                 + isComedy	+ isCrime +	isFantasy + isDocumentary	+ isFamily + isDrama + isHistory + isHorror + isMusic 
                 + isMystery + isRomance + isSciFi + isThriller + isSport + isSuperhero + isWar + isWestern + Duration_movie_final + Bugdet_final
                 + Actor_1_like_final + Actor_2_like_final + Actor_3_like_final + Director_like_final,
                 data = movie.train,
                 distribution = "gaussian",
                 n.trees = 1000,
                 shrinkage = 0.001,
                 interaction.depth = 2)

# NOTE: we need to specify number of trees to get a prediction for boosting
pred.boost <- predict(mod.boost, newdata = movie.test, n.trees=1000)
# after you have trained a model, you can get the predictions for all earlier iterations as well, for example:

pred.boost.earlier <- predict(mod.boost, newdata = movie.test, n.trees=330)

summary(mod.boost) # tells us what is the most influential variables.
#BUDGET seems to be the most influencial. region, year and main actor comes after. Duration and Genre doesnt seem to be influencial at all

# cross validation on n.trees and interaction depth
# since we have two tuning parameters, we need expand.grid
# WARNING: this took me ~1 hour to run
tGrid = expand.grid(n.trees = (1:75)*500, interaction.depth = c(1,2,4,6,8,10),
                    shrinkage = 0.001, n.minobsinnode = 10)

set.seed(232)
train.boost <- train(log(Gross_final) ~   Year + label_country + label_language + isAction
                     + isAdventure +	isAnimation	
                     + isBiography	
                     + isComedy	+ isCrime +	isFantasy + isDocumentary	+ isFamily + isDrama + isHistory + isHorror + isMusic 
                     + isMystery + isRomance + isSciFi + isThriller + isSport + isSuperhero + isWar + isWestern + Duration_movie_final + Bugdet_final
                     + Actor_1_like_final + Actor_2_like_final + Actor_3_like_final + Director_like_final,
                     data = movie.train,
                     method = "gbm",
                     tuneGrid = tGrid,
                     trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
                     metric = "RMSE",
                     distribution = "gaussian")
train.boost
# n.trees used is 6000 and interaction.depth = 10
best.boost <- train.boost$finalModel
test.mm = as.data.frame(model.matrix(log(Gross_final) ~   Year + label_country + label_language + isAction
                                     + isAdventure +	isAnimation	
                                     + isBiography	
                                     + isComedy	+ isCrime +	isFantasy + isDocumentary	+ isFamily + isDrama + isHistory + isHorror + isMusic 
                                     + isMystery + isRomance + isSciFi + isThriller + isSport + isSuperhero + isWar + isWestern + Duration_movie_final + Bugdet_final
                                     + Actor_1_like_final + Actor_2_like_final + Actor_3_like_final + Director_like_final, data=movie.test)) 
pred.best.boost <- predict(best.boost, newdata = test.mm, n.trees = 6000) # can use same model matrix

ggplot(train.boost$results, aes(x = n.trees, y = Rsquared, colour = as.factor(interaction.depth))) + geom_line() + 
  ylab("CV Rsquared") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) + 
  scale_color_discrete(name = "interaction.depth")

print("Boosting OSR2:")
OSR2(pred.best.boost, log(movie.test$Gross_final), log(movie.train$Gross_final))

print("Boosting Out-of-sample MAE:")
sum(abs(log(movie.test$Gross_final) - pred.best.boost))/nrow(test.mm)
