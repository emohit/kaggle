## Read training data
train=read.csv("Train.csv")
train[is.na(train)]=0
train.dat = train[-c(1:4)]

## Create baseline model
set.seed(43544)
baseline.model <- lm(YPLL.Rate~., data=train.dat, na.action= na.exclude)

## Read test data
test=read.csv("Test.csv")
test[is.na(test)]=0
test.dat = test[-c(1:4)]

## Score test data
pred <- predict(baseline.model, test.dat)


sample_submission=data.frame("ID"=test$ID,"Predicted"=pred)
write.csv(sample_submission,"Sample_submission.csv",row.names=F)
