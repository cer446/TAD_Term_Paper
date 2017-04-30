install.packages("data.table") 
library(data.table)
?fread

setwd("/Users/carolineroper/Documents/School/Text as Data")

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

colnames(train[0:5, 20:113])

# grow tree 
fit <- rpart(target ~ .,
             method="class", data=train[,20:113])

printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# plot tree 
plot(fit, uniform=TRUE, 
     main="Classification Tree for Kickstarter Data")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# create attractive postscript plot of tree 
post(fit, file = "/Users/carolineroper/Documents/School/Text as Data/tree.ps", 
     title = "Classification Tree for Kickstarter Data")
