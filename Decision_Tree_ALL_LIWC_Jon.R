rm(list = ls())
install.packages("ROCR")
install.packages("pRF")
install.packages("caret")
install.packages("randomForest")
install.packages("rpart") 
source("https://bioconductor.org/biocLite.R")
biocLite("multtest")
library(randomForest)
library(ROCR)
library(data.table)
library(caret)
library(pRF)
?fread

setwd("/Users/jonathanatoy/Desktop/kickstarter_project/TAD_Term_Paper")

Kickstarter_project_data_all <- fread("kickstarter_project_data_all.csv")

LIWC2015_Results_kickstarter <- fread("LIWC2015_Results_kickstarter.csv", na.strings=c("","NA", " "))

#&nbsp is also an html line, creates space (looks like we put this into the dictionary as well, so may want to re-run without that)
#remove underscores
#remove html tags between <> and that start with &

Kickstarter_project_data_all$description <- gsub("<[^>]*>","",Kickstarter_project_data_all$description)
Kickstarter_project_data_all$description <- gsub("&\\w+ ", "", Kickstarter_project_data_all$description)

column_names_first <- colnames(Kickstarter_project_data_all)
column_names_rest <- colnames(LIWC2015_Results_kickstarter[, 20:112])
column_names <- append(column_names_first, column_names_rest)
colnames(LIWC2015_Results_kickstarter) <- column_names

LIWC2015_Results_kickstarter <- LIWC2015_Results_kickstarter[-1, ]

#Remove rows with excessive missing values
#(note that these might be created by file reading problems, should see if we can find another solution)

LIWC2015_Results_kickstarter_c <- LIWC2015_Results_kickstarter[rowSums(is.na(LIWC2015_Results_kickstarter))<=5,]

#Save space by unloading dataset
rm(Kickstarter_project_data_all)
rm(LIWC2015_Results_kickstarter)
#Very few nulls remaining in the rows that aren't mostly null

View(LIWC2015_Results_kickstarter_c[rowSums(is.na(LIWC2015_Results_kickstarter_c))>=1,])

#Projects can be failed, canceled, successful or NA in rare cases

unique(LIWC2015_Results_kickstarter_c$project_final_state)

#NA's may represent projects for which the deadline hasn't passed

my.max <- function(x) ifelse( !all(is.na(x)), max(x, na.rm=T), NA)

my.max(LIWC2015_Results_kickstarter_c$project_deadline)

#5 entries that are relatively clean that have an NA final project state

View(LIWC2015_Results_kickstarter_c[is.na(LIWC2015_Results_kickstarter_c$project_final_state),])

#Received data on Mar 10th. Excluding projects with deadlines after Mar. 1st

LIWC2015_Results_kickstarter_c <- LIWC2015_Results_kickstarter_c[LIWC2015_Results_kickstarter_c$project_deadline < "2017-03-01"]

LIWC2015_Results_kickstarter_c$target <- ifelse(LIWC2015_Results_kickstarter_c$project_final_state=='successful', 1, 0)

#remove rows with NA for target (5 of them)
LIWC2015_Results_kickstarter_c <- LIWC2015_Results_kickstarter_c[complete.cases(LIWC2015_Results_kickstarter_c$target),]

#64,503 successes and 141,905 unsuccessful
table(LIWC2015_Results_kickstarter_c$target)

#The Random Forest function doesn't like there to be a feature called "function"
temp_col_names <- colnames(LIWC2015_Results_kickstarter_c)
temp_col_names[28] <- "function_"
colnames(LIWC2015_Results_kickstarter_c) <- temp_col_names

#Split Data

data(LIWC2015_Results_kickstarter_c)

## 75% of the sample size
smp_size <- floor(0.50 * nrow(LIWC2015_Results_kickstarter_c))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(LIWC2015_Results_kickstarter_c)), size = smp_size)

train <- LIWC2015_Results_kickstarter_c[train_ind, ]
test <- LIWC2015_Results_kickstarter_c[-train_ind, ]

#May do a learning curve analysis to figure out how much data we need/want
#Could also create dummies for project category and state (maybe city but this creates sparsity)

# Classification Tree with rpart
library(rpart)

colnames(train)

################
#Decision Trees#
################

# In below testing we can use the depth of the tree required to see even moderate improvements
# in the cost-complexity parameter as a proxy for the signal in the included variables. If there
# is no improvement at depth 30 (past this point the tree spews non-sense on 32-bit computers),
# can conclude that the variables alone are not enough to generate signal.

# Also note that the trees are being run without the WC variable, as it drowns out most other ones.

# grow tree using likely candidates (Excluding LIWC proprietary terms)
fit <- rpart(target ~ ., parms = list(split = "information"), control=list(cp=0.001),
             method="class", data=train[,c(50:51,58:59,64:65,80:84,95:97,99:100,113)])

printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# plot tree 
plot(fit, uniform=TRUE, 
     main="Classification Tree for Kickstarter Data")
text(fit, use.n=TRUE, all=TRUE, cex=.8)


# grow tree using only LIWC proprietary terms (No signal there)
fit_prop <- rpart(target ~ ., parms = list(split = "information"), control=list(cp=0.001),
             method="class", data=train[,c(21:24,113)])

printcp(fit_prop) # display the results 


# grow tree using only positive-negative emotions (Weak signal)
fit_emo <- rpart(target ~ ., parms = list(split = "information"), control=list(maxdepth=10,cp=0.0001),
                  method="class", data=train[,c(50:51,113)])

printcp(fit_emo) # display the results 
plotcp(fit_emo) # visualize cross-validation results 
summary(fit_emo) # detailed summary of splits
# plot tree 
plot(fit_emo, uniform=TRUE, 
     main="Classification Tree for Kickstarter Data")
text(fit_emo, use.n=TRUE, all=TRUE, cex=.8)

# grow tree using only risk/reward terms (Slightly stronger signal)
fit_rr <- rpart(target ~ ., parms = list(split = "information"), control=list(maxdepth = 5,cp=0.0001),
                 method="class", data=train[,c(80:81,113)])

printcp(fit_rr) # display the results 
plotcp(fit_rr) # visualize cross-validation results 
summary(fit_rr) # detailed summary of splits

# plot tree 
plot(fit_rr, uniform=TRUE, 
     main="Classification Tree for Kickstarter Data")
text(fit_rr, use.n=TRUE, all=TRUE, cex=.8)

# grow tree using only tentative/certain terms (No signal)
fit_tent <- rpart(target ~ ., parms = list(split = "information"), control=list(maxdepth = 30,cp=0.0001),
                method="class", data=train[,c(64:65,113)])
printcp(fit_tent) # display the results 


# grow tree using only informality terms (Moderate signal)
fit_inf <- rpart(target ~ ., parms = list(split = "information"), control=list(maxdepth = 10,cp=0.0001),
                 method="class", data=train[,c(95:97,99:100,113)])

printcp(fit_inf) # display the results 
plotcp(fit_inf) # visualize cross-validation results 
summary(fit_inf) # detailed summary of splits

# plot tree 
plot(fit_inf, uniform=TRUE, 
     main="Classification Tree for Kickstarter Data")
text(fit_inf, use.n=TRUE, all=TRUE, cex=.8)

# grow tree using only punctuation terms (Fairly strong signal)
fit_punct <- rpart(target ~ ., parms = list(split = "information"), control=list(maxdepth = 5,cp=0.0001),
                 method="class", data=train[,c(101:113)])

printcp(fit_punct) # display the results 
plotcp(fit_punct) # visualize cross-validation results 
summary(fit_punct) # detailed summary of splits
importance(fit_punct)
# plot tree 
plot(fit_punct, uniform=TRUE, 
     main="Classification Tree for Kickstarter Data")
text(fit_punct, use.n=TRUE, all=TRUE, cex=.8)

#Heavy use of colons, exclamations, apostraphes & dashes appear to signify greater project success
#Can likely link this to informality and relaxing of strict language norms (colloquial speak)


#####################
#Logistic Regression#
#####################

#Create a LM with all of the potential target variables

LR<-glm(target ~ ., family=binomial(link='logit'), data=subset(train,select=c(20:24,50:51,58:59,64:65,80:84,95:100,102:111,113)))
summary(LR)
anova(LR, test="Chisq")

fitted.results <- predict(LR,newdata=subset(test,select=c(20:24,50:51,58:59,64:65,80:84,95:100,102:111)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$target,na.rm = TRUE)
print(paste('Accuracy',1-misClasificError))


p <- predict(LR, newdata=subset(test,select=c(20:24,50:51,58:59,64:65,80:84,95:100,102:111)), type="response")
pr <- prediction(p, test$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Strip out punctuation and rerun
LR_no_punct<-glm(target ~ ., family=binomial(link='logit'), data=subset(train,select=c(20:24,50:51,58:59,64:65,80:84,95:100,113)))
summary(LR_no_punct)
anova(LR_no_punct, test="Chisq")

fitted.results <- predict(LR_no_punct,newdata=subset(test,select=c(20:24,50:51,58:59,64:65,80:84,95:100)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$target,na.rm = TRUE)
print(paste('Accuracy',1-misClasificError))

p <- predict(LR_no_punct, newdata=subset(test,select=c(20:24,50:51,58:59,64:65,80:84,95:100)), type="response")
pr <- prediction(p, test$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Use only informal categories
LR_inf<-glm(target ~ ., family=binomial(link='logit'), data=subset(train,select=c(20,95:100,102:111,113)))
summary(LR_inf)
anova(LR_inf, test="Chisq")

fitted.results <- predict(LR_inf,newdata=subset(test,select=c(20,95:100,102:111)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$target,na.rm = TRUE)
print(paste('Accuracy',1-misClasificError))

p <- predict(LR_inf, newdata=subset(test,select=c(20,95:100,102:111)), type="response")
pr <- prediction(p, test$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Use all linguistic features
LR_all<-glm(target ~ ., family=binomial(link='logit'), data=subset(train,select=c(20:113)))
summary(LR_all)


fitted.results <- predict(LR_all,newdata=subset(test,select=c(20:112)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$target,na.rm = TRUE)
print(paste('Accuracy',1-misClasificError))


p <- predict(LR_all, newdata=subset(test,select=c(20:112)), type="response")
pr <- prediction(p, test$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

###############
#Random Forest#
###############

#Notes that for RF, the AUC curves might need to be redone (they look too straight), 
#possibly using OOB estimate within the function

#Try RF using linguistic features (minus AllPunct, Period, Colon, Dash, Other)
fit=randomForest(factor(target) ~ ., na.action = na.omit, importance=TRUE, ntree=20, data=subset(train,select=c(20:100,102:103,105:107,109:110,113)))
(VI_F=round(importance(fit),2))

#Plot top variables wrt node impurity
varImpPlot(fit,type=1)
varImpPlot(fit,type=2)

p <- predict(fit, newdata=subset(test,select=c(20:100,102:103,105:107,109:110)), type="response")
pr <- prediction((p==1)*1, test$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Try RF with project category & funding goal(USD)
#Note: city & country had too many unique values for R to deal with as categorical

train$project_category <- as.factor(train$project_category)
train$project_city <- as.factor(train$project_city)
train$project_country <- as.factor(train$project_country)

test$project_category <- as.factor(test$project_category)
test$project_city <- as.factor(test$project_city)
test$project_country <- as.factor(test$project_country)


fit_meta=randomForest(factor(target) ~ ., na.action = na.omit, importance=TRUE, ntree=20, data=subset(train,select=c(5,13,20:100,102:103,105:107,109:110,113)))
(VI_F_meta=round(importance(fit_meta),2))

#Plot top variables wrt accuracy
varImpPlot(fit_meta,type=1)
#Plot top variables wrt node impurity
varImpPlot(fit_meta,type=2)

#Goal(USD) & project category have very large amount of signal, but linguistic
#features still have decently large importance (plus there's a lot more of them)


#Run permutation test on random forest to test significance of feature importance values
m <- floor(sqrt(length(c(5,13,20:100,102:103,105:107,109:110))))

p.test<-pRF(response=factor(train$target),
predictors=train[,c(5,13,20:100,102:103,105:107,109:110)],ntree = 20, mtry=m, n.perms=50,
type="classification",alpha=0.05)


sigplot(pRF.list=p.test,threshold=0.1)
varImpPlot(p.test$Model,1)

df<-cbind(p.test$Res.table,p.test$obs)
df[order(-df$MeanDecreaseGini),]
df[order(df$p.value, -df$MeanDecreaseGini),]
write.csv(df[order(df$p.value, -df$MeanDecreaseGini),], file = "pRF_feat_importance.csv")


p_rf <- predict(p.test$Model, newdata=subset(test,select=c(5,13,20:100,102:103,105:107,109:110)), type="prob")
pr_rf <- prediction(p_rf[,2], test$target)
prf_rf <- performance(pr_rf, measure = "tpr", x.measure = "fpr")
plot(prf_rf)
auc_rf <- performance(pr_rf, measure = "auc")
auc_rf <- auc_rf@y.values[[1]]
auc_rf
