
# Libraries: --------------------------------------------------------------
pacman::p_load(readr,corrplot, ggplot2, caret, highcharter, doParallel, parallel,
               plotly, caTools, dplyr,gdata, ranger, h2o, rstudioapi, plyr,
               RColorBrewer, randomForest, tidyr, forecast, lubridate, scatterplot3d)
################################################################################
#Importing Data
setwd("~/R studio/WIFI_project/Data.csv")

tdata <- read.csv("trainingData.csv", stringsAsFactors = FALSE)
vdata <- read.csv("validationData.csv", stringsAsFactors = FALSE)

############################## PRE- PROCESSING #################################
#Explorer data 
# summary(tdata[,521:529]) 
# summary(vdata[,521:529])


############## DATA CLEANING ###################
#Duplicates
sum(duplicated(tdata)) # sum of how many duplicates are in the data set 
sum(duplicated(vdata)) #0 

tdata <- distinct(tdata) #elimina filas duplicadas 
vdata <- distinct(vdata) #no tiene duplicados 


#MissingValues
sum(is.na(tdata)) #no missing values for both df
sum(is.na(vdata))

#EXPLORER DATA
# flat pLot training data  
plot(tdata$LONGITUDE, tdata$LATITUDE, xlab = "Longitude", ylab = "Latitude",
     main= "Trainig set - ubicactions")

# flat pLot VALIDATION data  
plot(vdata$LONGITUDE, vdata$LATITUDE, xlab = "Longitude", ylab = "Latitude",
     main= "Validation set - ubicactions")


#Checking distribution of Waps signal-----
# wapsTraining <- tdata[,1:520]
# 
# x <- wapsTraining
# x <- stack(x)
# 
# x <- x[-grep(0, x$values),]
hist(x$values, xlab = "WAP strength", main = "Distribution of WAPs signal strength (training set)", col = "red")

#validation
# wapsValidation <- vdata[,1:520]
# 
# x1 <- wapsValidation
# x1 <- stack(x1)
# 
# x1 <- x1[-grep(0, x1$values),]
hist(x1$values, xlab = "WAP strength", main = "Distribution of WAPs signal strength (validation set)", col = "red")


######################## TRANSFORMATION ###############################
#COMBINE DATASETS FOR SPEEDING UP THE PROCESS
ple_data <- gdata::combine(tdata, vdata) #"#F0E442",  "#0072B2"

plot_ly(ple_data, x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, color = ~source, 
        colors = c("#F0E442", "#0072B2"), size = 0.03) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = "Longitude"),
                      yaxis = list(title = "Latitude"),
                      zaxis = list(title = "Floor")))

factores <- c ("FLOOR", "BUILDINGID", "RELATIVEPOSITION", "SPACEID",
               "USERID", "PHONEID", "source")

ple_data[,factores] <- lapply(ple_data[,factores], as.factor)
rm(factores) # remueve objetos (todos los objets especificados)

numerics <- c("LONGITUDE", "LATITUDE")
ple_data[,numerics] <- lapply(ple_data[,numerics], as.numeric)
rm(numerics)



#EXPLORATION WAPS


#Change values of Waps=100 to -110 and organize all Waps in a vector 
Waps <- grep("WAP", names(ple_data), value =T) #save variables in a vector, vector solo con WAPS

ple_data[ple_data==100]<- -105
#a las variables numericas (waps) hago un cambio de variable afectando a la distribucion


#Use nearZeroVar() for removing such variables to save time during modeling.
Waps_train <- nearZeroVar(ple_data[ple_data$source=="tdata"
                                   ,Waps], saveMetrics = TRUE)

Waps_valid <- nearZeroVar(ple_data[ple_data$source=="vdata"
                                   ,Waps], saveMetrics = TRUE)


ple_data <- ple_data[-which(Waps_train$zeroVar == TRUE |
                              Waps_valid$zeroVar == TRUE)]


rm(Waps_train, Waps_valid)  #remueve objetos (todos los objets especificados)


#filter rows with RSSI signal equal to -105 (elimina filas que no emiten señal, =-105)
Waps <-grep("WAP", names(ple_data), value=T)

ple_data <- ple_data %>%
  filter(apply(ple_data[Waps], 1, function(x) length(unique(x))) > 1)



################ Training and Validation Set ###############################

#SEPARATE DATA INTO train AND valid SET 
split_data <- split(ple_data, ple_data$source)
list2env(split_data, envir = .GlobalEnv)
rm(split_data)

#Validation data to test performance
valid_data <- vdata

# Parallel Process
cluster <- makeCluster(detectCores() -1)
registerDoParallel(cluster)  

#Cross validation
cross_valid <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                            allowParallel = TRUE)


########################### Building prediction ##################################
h2o.init()
h2o.init(nthreads = -1, max_mem_size = "2G")
#h2o.removeAll()

#Pasamos los objetos a h20 y los nombramos para poder identificarlos. 
h2o_train <- as.h2o(tdata) 
h2o_valid <- as.h2o(valid_data)

Building_RF <- h2o.randomForest( y = "BUILDINGID", x = Waps,
                                 training_frame = h2o_train,
                                 nfolds = 8, ntrees = 100, max_depth = 30, seed = 123)

# # h2o.saveModel(object = Building_RF, path = getwd(), force = TRUE)
# modelo <- h2o.loadModel(path = "./DRF_model_R_1")
# modelo 
#save(Building_RF, file = "Building_RF.rda")
load("Building_RF.rda")


h2o.performance(Building_RF, h2o_valid) 

print(Building_RF) #accuracy. 1 

#####

Building_RF_Pred <- h2o.predict(Building_RF, h2o_valid) 

# Replacing original building id with predicted in validation dataset
ple_data$BUILDINGID[ple_data$source == "vdata"] <- as.data.frame(Building_RF_Pred)

#head(Building_RF_Pred)

######################### Predicting Longitud-Building #########################
#Se usa modelo Random Forest con procesamiento paralelo (no h20) 

building <- split(tdata, tdata$BUILDINGID)
list2env(building, envir = .GlobalEnv)

#Building 0 

# LongB0 <- randomForest(LONGITUDE~. -LATITUDE -FLOOR -SPACEID  -RELATIVEPOSITION -USERID -PHONEID -TIMESTAMP
#                        -BUILDINGID -source, data=building$"0", importance=T, maximize=T, method="rf",
#                        trControl=cross_valid, ntree=100, mtry=104,
#                        allowParalel=TRUE)

# save(LongB0, file = "LongB0.rda")
load("LongB0.rda")

LongB0_predict <- predict(LongB0, vdata[vdata$BUILDINGID == 0,])
LongB0_postresample <- postResample(LongB0_predict, 
                                    valid_data$LONGITUDE[valid_data$BUILDINGID == 0] )

LongB0_postresample #performance
#    RMSE    |   Rsquared   |     MAE 
# -----------|--------------|------------
# 7.0436205  |   0.9312404  | 4.6213629  


#Building 1 

# LongB1 <- randomForest(LONGITUDE~. -LATITUDE -FLOOR -SPACEID
#                        -RELATIVEPOSITION -USERID -PHONEID -TIMESTAMP
#                        -BUILDINGID -source, data=building$"1",
#                        importance=T, maximize=T, method="rf",
#                        trControl=cross_valid, ntree=100, mtry=104,
#                        allowParalel=TRUE)

# save(LongB1, file = "LongB1.rda")
load("LongB1.rda")

LongB1_predict <- predict(LongB1, vdata[vdata$BUILDINGID == 1,])
LongB1_postresample <- postResample(LongB1_predict, 
                                    valid_data$LONGITUDE[valid_data$BUILDINGID == 1] )
LongB1_postresample
#    RMSE    |   Rsquared   |     MAE 
# -----------|--------------|------------
# 9.1101534  |   0.9616834  | 6.5786404

#Building 2 

# LongB2 <- randomForest(LONGITUDE~. -LATITUDE -FLOOR -SPACEID
#                        -RELATIVEPOSITION -USERID -PHONEID -TIMESTAMP
#                        -BUILDINGID -source, data=building$"2",
#                        importance=T, maximize=T, method="rf",
#                        trControl=cross_valid, ntree=100, mtry=104,
#                        allowParalel=TRUE)

# save(LongB2, file = "LongB2.rda")
load("LongB2.rda")

LongB2_predict <- predict(LongB2, vdata[vdata$BUILDINGID == 2,])
LongB2_postresample <- postResample(LongB2_predict, 
                                    valid_data$LONGITUDE[valid_data$BUILDINGID == 2] )
LongB2_postresample
#    RMSE    |   Rsquared   |     MAE 
# -----------|--------------|------------
# 10.863392  |   0.8832729  | 7.1286247

#ALL PREDICTIONS BY LONGITUDE
total_LongPred <- c(LongB0_predict, LongB1_predict, LongB2_predict)


######################### Predicting Latitude-Building #########################
#Building 0 
# LattB0 <- randomForest(LATITUDE~. -LONGITUDE -FLOOR -SPACEID
#                        -RELATIVEPOSITION -USERID -PHONEID -TIMESTAMP
#                        -BUILDINGID -source, data=building$"0",
#                        importance=T, maximize=T, method="rf",
#                        trControl=cross_valid, ntree=100, mtry=104,
#                        allowParalel=TRUE)
# 
# save(LattB0, file = "LattB0.rda")
load("LattB0.rda")

LattB0_predict <- predict(LattB0, vdata[vdata$BUILDING == 0,])
LattB0_postresample <- postResample(LattB0_predict, valid_data$LATITUDE[valid_data$BUILDINGID ==  0])
LattB0_postresample
# RMSE        Rsquared       MAE 
# 5.5780701   0.9701076    3.8713950  

#Building 1 
# LattB1 <- randomForest(LATITUDE~. -LONGITUDE -FLOOR -SPACEID
#                        -RELATIVEPOSITION -USERID -PHONEID -TIMESTAMP
#                        -BUILDINGID -source, data=building$"1",
#                        importance=T, maximize=T, method="rf",
#                        trControl=cross_valid, ntree=100, mtry=104,
#                        alowParalel=TRUE)
#
#save(LattB1, file = "LattB1.rda")
load("LattB1.rda")

LattB1_predict <- predict(LattB1, vdata[vdata$BUILDING ==1,])
LattB1_postresample <- postResample(LattB1_predict, valid_data$LATITUDE[valid_data$BUILDINGID == 1])
LattB1_postresample

# RMSE          Rsquared        MAE 
# 10.7394524    0.9078965    7.4403477

#Building 2

# LattB2 <- randomForest(LATITUDE~. -LONGITUDE -FLOOR -SPACEID
#                        -RELATIVEPOSITION -USERID -PHONEID -TIMESTAMP
#                        -BUILDINGID -source, data=building$"2",
#                        importance=T, maximize=T, method="rf",
#                        trControl=cross_valid, ntree=100, mtry=104,
#                        alowParalel=TRUE)
# 
# save(LattB2, file = "LattB2.rda")
load("LattB2.rda")

LattB2_predict <- predict(LattB2, vdata[vdata$BUILDING ==2,])
LattB2_postresample <- postResample(LattB2_predict, valid_data$LATITUDE[valid_data$BUILDINGID == 2])
LattB2_postresample


# RMSE  Rsquared       MAE 
# 9.5760924 0.8912037 6.5701313 

#ALL PREDICTION BY LATITUDE
total_LattPred <- c(LattB0_predict, LattB1_predict, LattB2_predict)


########################### PREDICTING BY FLOOR ################################
#Add predictions about longitude y latitude for validation data
# ple_data$LONGITUDE [ple_data$source == "vdata"] <- total_LongPred
# ple_data$LATITUDE [ple_data$source == "vdata"] <- total_LattPred

# Split Data before modeling
ple_data_split<- split(ple_data, ple_data$source)
list2env(ple_data_split, envir=.GlobalEnv)
rm(ple_data_split)

# floor_longANDlatt <- randomForest(FLOOR~. -SPACEID
#                                   -RELATIVEPOSITION -USERID -PHONEID -TIMESTAMP
#                                   -BUILDINGID -source, data = tdata,
#                                   importance=T, maximize=T, method="rf",
#                                   trControl=cross_valid, ntree=100, mtry=35,
#                                   alowParalel=TRUE)
# 
# save(floor_longANDlatt, file = "floor_longANDlatt.rda")
load( "floor_longANDlatt.rda")

floor_predict <- predict(floor_longANDlatt, valid_data)
floor_pstresample <- postResample(floor_predict, valid_data$FLOOR )
floor_pstresample

# 
# Accuracy     Kappa 
# 0.9153915 0.8816471 


############## PREDICTIONS ALL  BUILDINGS BY long and lat ######################

# #PREDICT LONGITUDE BY ALL BUILDINGS

# Long_allBuildings <-randomForest(LONGITUDE~. -LATITUDE -FLOOR -SPACEID
#                   -RELATIVEPOSITION -USERID -PHONEID -BUIDLINGID
#                 -TIMESTAMP -source , data= tdata,
#                 importance=T,maximize=T, method="rf", trControl=cross_valid,
#                 ntree=100, mtry= 104, allowParalel=TRUE)
# 
#save(Long_allBuildings, file = "Long_allBuildings.rda")
load("Long_allBuildings.rda")

predict_Long_allBuildings <-predict(Long_allBuildings, valid_data)
postRes_Long_AllBRF<-postResample(predict_Long_allBuildings, valid_data$LONGITUDE)
postRes_Long_AllBRF

# RMSE         Rsquared        MAE 
# 11.0190509    0.9918934    7.0582954 


#PREDICT LATITTUDE BY ALL BUILDINGS
# Latt_allBuildings<-randomForest(LATITUDE~. -LONGITUDE -FLOOR -SPACEID -RELATIVEPOSITION -USERID -PHONEID
#                                  -TIMESTAMP -BUILDINGID -source , data= tdata,
#                            importance=T,maximize=T,
#                          method="rf", trControl=cross_valid,
#                                 ntree=100, mtry= 104,allowParalel=TRUE)

#save(Latt_allBuildings, file ="Latt_allBuildings.rda")
load("Latt_allBuildings.rda")


predict_Latt_allBuildings <-predict(Latt_allBuildings, valid_data)
postRes_Latt_AllB<-postResample(predict_Latt_allBuildings, valid_data$LATITUDE)
postRes_Latt_AllB

# RMSE  Rsquared       MAE 
# 9.5554487 0.9820718 6.1192459 


##################### ERRORS VISUALIZATION ###########################


#LONGITUDE 
Errors_Visual <-as.data.frame(cbind(predict_Long_allBuildings, valid_data$LONGITUDE))
Errors_Visual<-Errors_Visual %>% mutate(Error= Errors_Visual[,1]-Errors_Visual[,2])
Problems_LON<-Errors_Visual %>% filter(abs(Error)>10)


hist(Problems_LON$Error, 
     col="#0b87a1", xlab = "Error", ylab="Frequency",
     main="Error predicting Longitude")



# Mean Error
mean(Problems_LON$Error)#mean:4.507m
median(Problems_LON$Error)#median: 11.25m
MeanErrorLong<-mean(abs(Problems_LON$Error))  

#LATITUDE  

Errors_Visual_LATT <- as.data.frame(cbind(predict_Latt_allBuildings, valid_data$LONGITUDE))
Errors_Visual_LATT<-Errors_Visual_LATT %>% mutate(Error= Errors_Visual_LATT[,1]-Errors_Visual_LATT[,2])
Problems_LATT<-Errors_Visual_LATT %>% filter(abs(Error)>10)

hist(Problems_LATT$Error, 
     col="#0b87a1", xlab = "Error", ylab="Frequency",
     main="Error predicting Latitude")


# Mean Error
mean(Problems_LATT$Error)
median(Problems_LATT$Error)
MeanErrorLatt <-mean(abs(Problems_LATT$Error))  



########## VISUALIZATION PREDICT VS REAL POINTS #####################

total_All_predictions <- data.frame(total_LongPred, total_LattPred,  floor_predict, as.data.frame(Building_RF_Pred[1]) )

#Change names of each columns
# names(total_All_predictions)[1] <- "LONGITUDE"
# names(total_All_predictions)[2] <- "LATITUDE"
# names(total_All_predictions)[3] <- "FLOOR"
# names(total_All_predictions)[4] <- "BUILDINGID"
# total_All_predictions["source"] <- "Predicted"
# total_All_predictions$source <- as.factor(total_All_predictions$source)
# total_All_predictions$FLOOR <- as.numeric(total_All_predictions$FLOOR)


real_values <- valid_data[c("LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID")]
real_values["source"] <- "Real"
real_values$source <- as.factor(real_values$source)
real_values$FLOOR <- as.numeric(real_values$FLOOR)


final_visualization <- rbind(total_All_predictions, real_values)
# rm(total_All_predictions)
# rm(real_values)

plot_ly(final_visualization, x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, color = ~source, colors = c("#0072B2", "#F0E442"), size = 0.1) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Longitude'),
                      yaxis = list(title = 'Latitude'),
                      zaxis = list(title = 'Floor')))
 
plot(LATITUDE ~ LONGITUDE, data = valid_data, pch = 20, col = "grey53")
points(LATITUDE ~ LONGITUDE, data = vdata, pch = 20, col = "blue")



#Importing Data
setwd("~/R studio/WIFI_project/Data.csv")

test_data <- read.csv("testData.csv", stringsAsFactors = FALSE)

#Pre-processing
#test_data <- distinct(test_data)
# 
Waps <- grep("WAP", names(test_data), value =T) #save variables in a vector, vector solo con WAPS
test_data[test_data==100]<- -105


#-----------------------------------------------------------------------------------------------

# h2o.init(nthreads = -1, max_mem_size = "2G")
# h2o.removeAll()

#load("Building_RF.rda")

# h2o.performance(Building_RF, test_h2o)
# Building_RF_Pred <- h2o.predict(Building_RF, test_h2o)
# h2o.shutdown()



#### TESTING #######

test_h2o <- as.h2o(test_data)

# Building_model <- h2o.predict(Building_RF, test_h2o)
# h2o.performance(Building_RF, test_h2o)
test_data$BUILDINGID <- as.data.frame(Building_model)$predict

test_data$BUILDINGID <- as.character(test_data$BUILDINGID)

test_data$BUILDINGID <- as.factor(test_data$BUILDINGID)
building_set <- split(test_data, test_data$BUILDINGID)

list2env(building_set, envir = .GlobalEnv)


###### Longitude Predictions ####### 
long_predictions_B0 <- predict(LongB0, test_data[test_data$BUILDING=="0",])
building_set$"0"$LONGITUDE <- long_predictions_B0 

long_predictions_B1 <- predict(LongB1, test_data[test_data$BUILDINGID == "1", ])
building_set$"1"$LONGITUDE <- long_predictions_B1

long_predictions_B2 <- predict(LongB2, test_data[test_data$BUILDINGID == "2", ])
building_set$"2"$LONGITUDE <- long_predictions_B2

longBuilding_pred <- c(long_predictions_B0, long_predictions_B1,long_predictions_B2 )

###### Latitude Predictions ####### 

latt_predictions_B0 <- predict(LattB0, test_data[test_data$BUILDING=="0",])
building_set$"0"$LATITUDE <- latt_predictions_B0 

latt_predictions_B1 <- predict(LattB1, test_data[test_data$BUILDINGID == "1", ])
building_set$"1"$LATITUDE <- latt_predictions_B1

latt_predictions_B2 <- predict(LattB2, test_data[test_data$BUILDINGID == "2", ])
building_set$"2"$LATITUDE <- latt_predictions_B2

lattBuilding_pred <- c(latt_predictions_B0, latt_predictions_B1, latt_predictions_B2)

#################################################################################

test_data$LONGITUDE <- longBuilding_pred
test_data$LATITUDE <- lattBuilding_pred

testing_set <- bind_rows(building_set)

floor_pred <- predict (floor_longANDlatt, testing_set)

testing_set$FLOOR <- floor_pred


######################################################################
#3D Plot for comparing real and predicted points: ----
final_predictions <- data.frame(LONGITUDE = testing_set$LONGITUDE,LATITUDE = testing_set$LATITUDE, 
                                FLOOR = testing_set$FLOOR, BUILDINGID = testing_set$BUILDINGID )


final_predictions$FLOOR <- as.numeric(final_predictions$FLOOR)

plot_ly(final_predictions, x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, color = ~BUILDINGID, size = 0.01) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Longitude'),
                      yaxis = list(title = 'Latitude'),
                      zaxis = list(title = 'Floor')))


test_set_predictions <- testing_set[c("LATITUDE", "LONGITUDE", "FLOOR")]

write.csv(test_set_predictions, "KeylaRandomForest.csv", quote = FALSE,
          row.names = FALSE)








