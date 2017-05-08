rm(list = ls())
#install.packages("ROCR")
#install.packages("caret")
#install.packages("randomForest")
#install.packages("rpart") 
library(randomForest)
library(ROCR)
library(data.table)
library(caret)
?fread

#setwd("/Users/jonathanatoy/Desktop/kickstarter_project/TAD_Term_Paper")
setwd("/Users/carolineroper/Documents/School/Text as Data")

Kickstarter_project_data_all <- fread("kickstarter_project_data_all.csv")

LIWC2015_Results_kickstarter <- fread("LIWC2015_Results_kickstarter.csv", na.strings=c("","NA", " "))

#&nbsp is also an html line, creates space (looks like we put this into the dictionary as well, so may want to re-run without that)
#remove html tags between <> and that start with &
#these preprocessing steps don't matter unless we do them before we apply the dictionaries.

#may want to remove underscores


Kickstarter_project_data_all$description <- gsub("<[^>]*>","",Kickstarter_project_data_all$description)
Kickstarter_project_data_all$description <- gsub("&\\w+ ", "", Kickstarter_project_data_all$description)

column_names_first <- colnames(Kickstarter_project_data_all)
column_names_rest <- colnames(LIWC2015_Results_kickstarter[, 20:112])
column_names <- append(column_names_first, column_names_rest)
colnames(LIWC2015_Results_kickstarter) <- column_names

LIWC2015_Results_kickstarter <- LIWC2015_Results_kickstarter[-1, ]

View(LIWC2015_Results_kickstarter)

#Remove rows with excessive missing values
#(note that these might be created by file reading problems, should see if we can find another solution)

#Analyze data cleaning
#only 598 messy entries out of 208,827 total that have this issue - so unlikely to change our overall results
View(LIWC2015_Results_kickstarter[rowSums(is.na(LIWC2015_Results_kickstarter))>5,])
nrow(LIWC2015_Results_kickstarter)
#Very few nulls remaining in the rows that aren't mostly null
#View(LIWC2015_Results_kickstarter[rowSums(is.na(LIWC2015_Results_kickstarter_c))>=1,])
#Projects can be failed, canceled, successful or NA in rare cases
unique(LIWC2015_Results_kickstarter$project_final_state)
#5 entries that are relatively clean that have an NA final project state
#View(LIWC2015_Results_kickstarter_c[is.na(LIWC2015_Results_kickstarter_c$project_final_state),])

#Perform data cleaning
LIWC2015_Results_kickstarter_c <- LIWC2015_Results_kickstarter[rowSums(is.na(LIWC2015_Results_kickstarter))<=5,]

#Save space by unloading dataset

my.max <- function(x) ifelse( !all(is.na(x)), max(x, na.rm=T), NA)
my.max(LIWC2015_Results_kickstarter_c$project_deadline)

#Received data on Mar 10th. Excluding projects with deadlines after Mar. 1st

LIWC2015_Results_kickstarter_c <- LIWC2015_Results_kickstarter_c[LIWC2015_Results_kickstarter_c$project_deadline < "2017-03-01"]
LIWC2015_Results_kickstarter_c$target <- ifelse(LIWC2015_Results_kickstarter_c$project_final_state=='successful', 1, 0)

#remove rows with NA for target (5 of them)
LIWC2015_Results_kickstarter_c <- LIWC2015_Results_kickstarter_c[complete.cases(LIWC2015_Results_kickstarter_c$target),]

#64,504 successes and 141,905 unsuccessful
table(LIWC2015_Results_kickstarter_c$target)

#The Random Forest function doesn't like there to be a feature called "function"
temp_col_names <- colnames(LIWC2015_Results_kickstarter_c)
temp_col_names[28] <- "function_"
colnames(LIWC2015_Results_kickstarter_c) <- temp_col_names

#Split Data

data(LIWC2015_Results_kickstarter_c)

## 50% of the sample size
smp_size <- floor(0.50 * nrow(LIWC2015_Results_kickstarter_c))

## set the seed to make partition reproductible
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

colnames(train[,c(50:51,58:59,64:65,80:84,95:97,99:100,113)])

# Also note that the trees are being run without the WC variable, as it drowns out most other ones.

# grow tree using likely candidates (Excluding LIWC proprietary terms)
fit <- rpart(target ~ ., parms = list(split = "information"), control=list(cp=0.001),
             method="class", data=train[,c(50:51,58:59,64:65,80:84,95:97,99:100,113)]) #new list of columns, cp is "complexity parameter", 
#can also look at minsplit, minbucket, and maxdepth, currently not being used

?rpart

printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# plot tree 
plot(fit, uniform=TRUE, 
     main="Classification Tree for Kickstarter Data")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

colnames(train[,c(21:24,113)])

# grow tree using only LIWC proprietary terms (No signal there)
fit_prop <- rpart(target ~ ., parms = list(split = "information"), control=list(cp=0.001),
             method="class", data=train[,c(21:24,113)])

printcp(fit_prop) # display the results 

colnames(train[,c(50:51,113)])

# grow tree using only positive-negative emotions (Weak signal, not enough to grow tree)
fit_emo <- rpart(target ~ ., parms = list(split = "information"), control=list(maxdepth=10,cp=0.0001),
                  method="class", data=train[,c(50:51,113)])

printcp(fit_emo) # display the results 
plotcp(fit_emo) # visualize cross-validation results 
summary(fit_emo) # detailed summary of splits
# plot tree 
plot(fit_emo, uniform=TRUE, 
     main="Classification Tree for Kickstarter Data")
text(fit_emo, use.n=TRUE, all=TRUE, cex=.8)

colnames(train[,c(80:81,113)])

max(train$risk)

min(train$risk)

mean(train$risk)

median(train$risk)

# grow tree using only risk/reward terms (Slightly stronger signal)
fit_rr <- rpart(target ~ ., parms = list(split = "information"), control=list(maxdepth = 5,cp=0.0001),
                 method="class", data=train[,c(80:81,113)])

?rpart
printcp(fit_rr) # display the results 
plotcp(fit_rr) # visualize cross-validation results 
summary(fit_rr) # detailed summary of splits

to_plot <- test[,c(80:81,113)]

to_plot$target <-test$target
to_plot$predict_prob <- predict(fit_rr, test)[,2]

unique(to_plot$predict_prob)

to_plot$reward_rounded <- round(to_plot$reward, digits=1)
to_plot$risk_rounded <- round(to_plot$risk, digits=1)

by_attr <- group_by(to_plot, reward_rounded, risk_rounded)
by_attr <- dplyr::summarise(by_attr,
                            count = n(),
                            successful=sum(target),
                            percent_successful = sum(target)/n(),
                            prediction = mean(predict_prob)
)

colnames(by_attr) <- c('reward_rounded', 'risk_rounded', 'count', 'successful', 'Percent_Successful', 'Prediction')

#should I create these plots on the test data or the train data?
#only reason to do it on training data is the idea of not looking at test data until you have a final model

ggplot(by_attr, aes(reward_rounded, risk_rounded)) + geom_tile(aes(fill = Percent_Successful), colour = "white") +
  scale_fill_gradient(low = "red", high = "steelblue") +
  theme(panel.background = element_rect(fill="white")) +
  labs(title = "Percent Successful by Risk and Reward") +
  scale_x_continuous(limits = c(0,5)) +
  scale_y_continuous(limits = c(0,5)) +
  xlab("Reward") +
  ylab('Risk') +
  labs(fill="Percent Successful")

View(by_prediction)

ggplot(by_attr, aes(reward_rounded, risk_rounded)) + geom_tile(aes(fill = Prediction), colour = "white") +
  scale_fill_gradient(low = "red", high = "steelblue") +
  labs(title = "Predicted Probability of Success in Terms of Risk and Reward") +
  theme(panel.background = element_rect(fill="white")) +
  scale_x_continuous(limits = c(0, 5)) +
  scale_y_continuous(limits = c(0, 5)) +
  xlab("Reward") +
  ylab('Risk') +
  labs(fill="Success Likelihood")

# plot tree 
plot(fit_rr, uniform=TRUE, 
     main="Classification Tree for Kickstarter Data", cex =.5)
text(fit_rr, use.n=TRUE, all=TRUE, cex=.5)

post(fit, file = "/Users/carolineroper/Documents/School/Text as Data/risk-reward-tree.ps", 
     title = "Classification Tree for Kickstarter Data")

# grow tree using only tentative/certain terms (No signal)
colnames(train[,c(64:65,113)])

fit_tent <- rpart(target ~ ., parms = list(split = "information"), control=list(maxdepth = 30,cp=0.0001),
                method="class", data=train[,c(64:65,113)])
printcp(fit_tent) # display the results 

colnames(train[,c(95:97,99:100,113)])

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

colnames(train[,c(101:113)])

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

# grow tre using only past/present/future (not enough signal to grow the tree)

which( colnames(train)=="focuspast" )

#82,83,84

View(train)

fit_time <- rpart(target ~ ., parms = list(split = "information"), control=list(maxdepth = 5,cp=0.0001),
                   method="class", data=train[,c(82,83,84,113)])

printcp(fit_time) # display the results 
plotcp(fit_time) # visualize cross-validation results 
summary(fit_time) # detailed summary of splits
importance(fit_time)
# plot tree 
plot(fit_punct, uniform=TRUE, 
     main="Classification Tree for Kickstarter Data")
text(fit_punct, use.n=TRUE, all=TRUE, cex=.8)



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
varImpPlot(fit,type=2)

p <- predict(fit, newdata=subset(test,select=c(20:100,102:103,105:107,109:110)), type="response")
pr <- prediction((p==1)*1, test$target)
pr <- prediction((p==1)*1, test$target, type="prob")
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

?prediction

p <- predict(fit_meta, newdata=subset(test,select=c(5,13,20:100,102:103,105:107,109:110)), type="prob")
pr <- prediction(p[,2], test$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

which( colnames(train)=="focusfuture" )
which( colnames(train)=="focuspresent" )

to_plot <- test[,c(83:84,113)]

#to_plot$target <-test$target
to_plot$predict_prob <- p[,2]

unique(to_plot$predict_prob)

to_plot$focusfuture_rounded <- round(to_plot$focusfuture, digits=1)
to_plot$focuspresent_rounded <- round(to_plot$focuspresent, digits=1)

by_attr <- group_by(to_plot, focusfuture_rounded, focuspresent_rounded)
by_attr <- dplyr::summarise(by_attr,
                            count = n(),
                            successful=sum(target),
                            percent_successful = sum(target)/n(),
                            prediction = mean(predict_prob)
)

#should I create these plots on the test data or the train data?
#only reason to do it on training data is the idea of not looking at test data until you have a final model

ggplot(by_attr, aes(focusfuture_rounded, focuspresent_rounded)) + geom_tile(aes(fill = percent_successful), colour = "white") +
  scale_fill_gradient(low = "red", high = "steelblue") +
  theme(panel.background = element_rect(fill="white")) +
  labs(title = "Percent Successful by Focus Future and Focus Present") +
  scale_x_continuous(limits = c(0,7)) +
  scale_y_continuous(limits = c(0,15)) +
  xlab("Future") +
  ylab("Present") +
  labs(fill="Percent Successful")

View(by_prediction)

ggplot(by_attr, aes(focusfuture_rounded, focuspresent_rounded)) + geom_tile(aes(fill = prediction), colour = "white") +
  scale_fill_gradient(low = "red", high = "steelblue") +
  labs(title = "Predicted Probability of Success in Terms of Future and Present Focus") +
  theme(panel.background = element_rect(fill="white")) +
  scale_x_continuous(limits = c(0,7)) +
  scale_y_continuous(limits = c(0,15)) +
  xlab("Future") +
  ylab("Present") +
  labs(fill="Success Likelihood")

