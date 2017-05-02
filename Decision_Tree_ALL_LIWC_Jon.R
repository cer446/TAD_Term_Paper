install.packages("rpart") 
library(data.table)
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

#Very few nulls remaining in the rows that aren't mostly null

View(LIWC2015_Results_kickstarter_c[rowSums(is.na(LIWC2015_Results_kickstarter_c))>=1,])

#Projects can be failed, canceled, successful or NA in rare cases

unique(LIWC2015_Results_kickstarter_c$project_final_state)

#NA's may represent projects for which the deadline hasn't passed

my.max <- function(x) ifelse( !all(is.na(x)), max(x, na.rm=T), NA)

my.max(LIWC2015_Results_kickstarter$project_deadline)

#5 entries that are relatively clean that have an NA final project state

View(LIWC2015_Results_kickstarter_c[is.na(LIWC2015_Results_kickstarter_c$project_final_state),])

#Received data on Mar 10th. Excluding projects with deadlines after Mar. 1st

LIWC2015_Results_kickstarter_c <- LIWC2015_Results_kickstarter_c[LIWC2015_Results_kickstarter_c$project_deadline < "2017-03-01"]

LIWC2015_Results_kickstarter_c$target <- ifelse(LIWC2015_Results_kickstarter_c$project_final_state=='successful', 1, 0)

#64,503 successes and 141,905 unsuccessful

table(LIWC2015_Results_kickstarter_c$target)

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

colnames(train[0:5, c(21:24,50:51,58:59,64:65,80:84,95:97,99:100,113)])

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

# plot tree 
plot(fit_punct, uniform=TRUE, 
     main="Classification Tree for Kickstarter Data")
text(fit_punct, use.n=TRUE, all=TRUE, cex=.8)

#Heavy use of colons, exclamations, apostraphes & dashes appear to signify greater project success
#Can likely link this to informality and relaxing of strict language norms (colloquial speak)


# create attractive postscript plot of tree 
#post(fit, file = "tree.ps", 
#     title = "Classification Tree for Kickstarter Data")

