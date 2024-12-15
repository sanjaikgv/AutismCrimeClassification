# Load necessary libraries
library(e1071)       # For SVM
library(caret)       # For data splitting and evaluation
library(dplyr)       # For data manipulation
library(purrr)       # For oversampling training data
library(ggplot2)     # For visualization (optional)
library(pROC)        # For evaluating ROC and calculating AUC
library(doParallel)  #Parallel Processing
library(foreach)     #Parallel processiong

# Import the dataset
file_path = "/Users/sanjaikgv/Downloads/cleaningMerged2.csv" 
data = read.csv(file_path)

# Preview the data
str(data)

#Renaming the variables
data = data %>%
  rename(
    ID = X,
    CaseReportNumber = Case.Report.Number,
    OccurredDate = Occurred.Date,
    OccurredDay = Occurred.Day.of.week,
    OccurredTime = Occurred.Time,
    CouncilDistrict = Council.District,
    CensusBlockGroup = Census.Block.Group,
    Sector = Sector,
    ZipCode = Zip.Code,
    OffenseDescription = Highest.Offense.Description,
    NIBRSCategory = NIBRS.Category,
    NIBRSDesc = NIBRS.Description,
    NIBRSGroup = NIBRS.Group,
    OccurredYear = Occurred.Year,
    OccurredMonth = Occurred.Month,
    ReportedDate = Reported.Date,
    LocationDescription = Location.Description,
    VictimGender = Victim.Gender,
    VictimAgeRange = Victim.Age.Range,
    VictimRaceEthnicity = Victim.Race.Ethnicity,
    NIBRSCodeDescription = NIBRS.Offense.Code.and.Extension.Description,
    InternalClearanceStatus = Internal.Clearance.Status,
    OffenseCode = Highest.Offense.Code,
    FamilyViolence = Family.Violence,
    OccurredDateTime = Occurred.Date.Time,
    ReportDateTime = Report.Date.Time,
    ReportDate = Report.Date,
    ReportTime = Report.Time,
    LocationType = Location.Type,
    APDSector = APD.Sector,
    APDDistrict = APD.District,
    ClearanceStatus = Clearance.Status,
    ClearanceDate = Clearance.Date,
    UCRCategory = UCR.Category,
    CategoryDescription = Category.Description,
    OccurredHour = Occurred.Hour,
    OtherNIBRSCategory = other_NIBRS.Category,
    OtherNIBRSDetail = other_NIBRS.Description,
    OtherNIBRSGroup = other_NIBRS.Group,
    OtherLocationDescription = other_Location.Description,
    OtherVictimGender = other_Victim.Gender,
    OtherVictimAgeRange = other_Victim.Age.Range,
    OtherVictimRaceEthnicity = other_Victim.Race.Ethnicity,
    OtherNIBRSCodeDescription = other_NIBRS.Offense.Code.and.Extension.Description
  )

#Dropping few irrelevant variables
data = data %>%
  select(-c(
    ID,
    CaseReportNumber,
    ReportedDate,
    ReportDate,
    ReportDateTime,
    ReportTime,
    ClearanceStatus,
    ClearanceDate,
    InternalClearanceStatus,
    OccurredDateTime,
    OtherNIBRSCategory, 
    OtherNIBRSDetail, 
    OtherNIBRSGroup,
    OtherLocationDescription, 
    OtherVictimGender,
    OtherVictimAgeRange,
    OtherVictimRaceEthnicity,
    OtherNIBRSCodeDescription
  ))

#Extracting date column from the OccurredDate 
data$OccurredFullDate = data$OccurredDate
data$OccurredDate = as.factor(format(as.Date(data$OccurredDate, "%Y-%m-%d"), "%d"))

#Reorganizing the predictors 
new_data = data %>%
  select(c(
    # Time Variables - 6
    OccurredFullDate,
    OccurredYear,
    OccurredMonth,
    OccurredDate,
    OccurredDay,
    OccurredHour,
    # Location Variables - 6
    APDDistrict,
    Sector,
    LocationDescription,
    CouncilDistrict,
    CensusBlockGroup,
    ZipCode,
    # Offense Variables - 5
    NIBRSCategory,
    NIBRSDesc,
    NIBRSGroup,
    NIBRSCodeDescription,
    OffenseCode,
    # Victim Variables - 4
    VictimGender,
    VictimAgeRange,
    VictimRaceEthnicity,
    FamilyViolence
  ))

# Viz1
class_counts = table(data$NIBRSDesc)
class_percentage = prop.table(class_counts) * 100
class_data = data.frame(Class = names(class_counts), 
                        Count = as.vector(class_counts), 
                        Percentage = as.vector(class_percentage))

#checking levels on NIBRSDesc
table(new_data$NIBRSDesc)

NIBRS_Desc_to_remove = c("09B Homicide: Negligent Manslaughter",
                         "09C Homicide: Justifiable Homicide (NOT A CRIME)",
                         "36A Sex Offense","36B Sex Offense",
                         "64A Human Trafficking","64B Human Trafficking")

# Create the bar chart
ggplot(class_data, aes(x = Class, y = Count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = sprintf("%.1f%%", Percentage)), vjust = -0.5, size = 3.5) +
  labs(title = "NIBRSDescription Class Distribution", x = "Classes", y = "Count") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


#Removing underrepresented levels (<100)
new_data <- new_data %>% 
  filter(!NIBRSDesc %in% NIBRS_Desc_to_remove)

# Converting the predictors to factors for SVM training 
new_data = new_data %>%
  mutate(
    OccurredYear = as.factor(OccurredYear),       
    OccurredMonth = as.factor(OccurredMonth),     
    OccurredDate = as.factor(OccurredDate),       
    OccurredDay = as.factor(OccurredDay),         
    OccurredHour = as.factor(OccurredHour),     
    APDDistrict = as.factor(APDDistrict),         
    Sector = as.factor(Sector),                   
    LocationDescription = as.factor(LocationDescription), 
    CouncilDistrict = as.factor(CouncilDistrict), 
    CensusBlockGroup = as.factor(CensusBlockGroup), 
    ZipCode = as.factor(ZipCode),                 
    NIBRSCategory = as.factor(NIBRSCategory),     
    NIBRSDesc = as.factor(NIBRSDesc),             
    NIBRSGroup = as.factor(NIBRSGroup),           
    NIBRSCodeDescription = as.factor(NIBRSCodeDescription), 
    OffenseCode = as.factor(OffenseCode),         
    VictimGender = as.factor(VictimGender),       
    VictimAgeRange = as.factor(VictimAgeRange),   
    VictimRaceEthnicity = as.factor(VictimRaceEthnicity), 
    FamilyViolence = as.factor(FamilyViolence),    
    OccurredFullDate = as.factor(OccurredFullDate)
  )


#Choosing Predictors based on results obtained from tree based and ensemble methods
my_data = new_data %>%
  select(c(OccurredMonth, OccurredDay, OccurredHour,
           ZipCode, LocationDescription,
           VictimGender, VictimAgeRange, VictimRaceEthnicity,
           NIBRSDesc))

#data split
set.seed(3)
sample_index = sample(seq_len(nrow(my_data)), size = 0.8 * nrow(my_data))
train_data = my_data[sample_index, ]
test_data = my_data[-sample_index, ]


#---------- Linear SVM Fit ---------------
#----- Setting Parallel Processing without hyperparameter tuning -------
n_cores = detectCores() - 3  
cl = makeCluster(n_cores)
registerDoParallel(cl)

train_control = trainControl(
  method = "cv",            # Cross-validation
  number = 5,               # Number of folds for CV
  allowParallel = TRUE      # Enable parallel processing
)

l_tune_grid <- expand.grid(
  C = c(1)      
)

#----- Run Model -------
linear_svm_model = train(
  NIBRSDesc ~ .,             
  data = train_data,         
  method = "svmLinear",      
  trControl = train_control, 
  tuneGrid = l_tune_grid,   
  verbose = TRUE
)

linear_svm = readRDS("linear_svm_model.rds")

#evaluate error
predictions_linear_svm = predict(linear_svm, newdata = test_data) 

levels(predictions_linear_svm)

l_conf_matrix = confusionMatrix(as.factor(predictions_linear_svm), as.factor(test_data$NIBRSDesc))
l_conf_matrix$byClass

l_conf_matrix_df = as.data.frame(l_conf_matrix$table)

#Linear Kernel Confusion Matrix Heatmap Visualization
ggplot(l_conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Confusion Matrix Heatmap - Linear Kernel without CV",
       x = "Predicted Class", y = "Actual Class") +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

l_svm_roc_curve = multiclass.roc(test_data$NIBRSDesc, as.numeric(predictions_linear_svm))
l_svm_auc_score = auc(l_svm_roc_curve)

l_svm_auc_score

#----- Setting Parallel Processing with hyperparameter tuning -------
n_cores = detectCores() - 3  
cl = makeCluster(n_cores)
registerDoParallel(cl)

train_control = trainControl(
  method = "cv",            # Cross-validation
  number = 5,               # Number of folds for CV
  allowParallel = TRUE      # Enable parallel processing
)

l_tune_grid <- expand.grid(
  C = c(0.1,1,10)      # Values for the cost parameter
)

#----- Run Model -------
linear_svm_model = train(
  NIBRSDesc ~ .,             
  data = train_data,         
  method = "svmLinear",      
  trControl = train_control, 
  tuneGrid = l_tune_grid,
  verbose = TRUE
)

saveRDS(linear_svm_model, "linear_cv_svm_model.rds")

stopCluster(cl)

cv_linear_svm = readRDS("linear_cv_svm_model.rds")

predictions_cv_linear_svm = predict(cv_linear_svm, newdata = test_data)

l_cv_conf_matrix = confusionMatrix(as.factor(predictions_cv_linear_svm), as.factor(test_data$NIBRSDesc))

l_cv_conf_matrix_df = as.data.frame(l_cv_conf_matrix$table)

#Linear Kernel Confusion Matrix Heatmap Visualization
ggplot(l_conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Confusion Matrix Heatmap - Linear Kernel with CV", x = "Predicted Class", y = "Actual Class") +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

l_cv_svm_roc_curve = multiclass.roc(test_data$NIBRSDesc, as.numeric(predictions_cv_linear_svm))
l_cv_svm_auc_score = auc(l_cv_svm_roc_curve)


#---------- Radial SVM Fit ---------------
#----- Setting Parallel Processing -------
n_cores = detectCores() - 3  
cl = makeCluster(n_cores)
registerDoParallel(cl)

train_control = trainControl(
  method = "none",          
  allowParallel = TRUE      # Enable parallel processing
)

r_tune_grid <- expand.grid(
  C = c(1),       
  sigma = c(0.1)
)

#----- Run Model -------
radial_svm_model = train(
  NIBRSDesc ~ .,             
  data = train_data,        
  method = "svmRadial",      
  trControl = train_control, 
  tuneGrid = r_tune_grid,    
  verbose = TRUE
)

saveRDS(radial_svm_model, "radial_svm_model.rds")

stopCluster(cl)

#radial_svm = svm(NIBRSDesc ~ ., data = train_data, kernel = "radial", 
#                 cost = 1, gamma = 0.1, scale = FALSE)

radial_svm = readRDS("radial_svm_model.rds")

#evaluate error
predictions_radial_svm = predict(radial_svm, newdata = test_data) 
confusionMatrix(predictions_radial_svm, test_data$NIBRSDesc)

r_conf_matrix = confusionMatrix(as.factor(predictions_radial_svm), as.factor(test_data$NIBRSDesc))
r_conf_matrix$byClass

r_conf_matrix_df = as.data.frame(r_conf_matrix$table)

ggplot(r_conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Confusion Matrix Heatmap - Radial Kernel", x = "Predicted Class", y = "Actual Class") +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

r_svm_roc_curve = multiclass.roc(test_data$NIBRSDesc, as.numeric(predictions_radial_svm))
r_svm_auc_score = auc(r_svm_roc_curve)

#----One vs Rest Radial Kernel Fitting-----------
ovr_train_data = train_data
ovr_classes = unique(train_data$NIBRSDesc)

models_ovr = list()

n_cores = detectCores() - 2  # Leave one core for other tasks
cl = makeCluster(n_cores)
registerDoParallel(cl)

Sys.time()
models_ovr = foreach(class = ovr_classes, .packages = c("e1071", "dplyr"),
                     .combine = 'c') %dopar% {
  
  ovr_train_data$binary_label = ifelse(ovr_train_data$NIBRSDesc == class, 1, 0)
  
  # Train the SVM model for each class
  model_ovr = svm(binary_label ~ .-NIBRSDesc, data = ovr_train_data, 
              kernel = "radial", cost = 1, sigma = 0.1, gamma = 0.1)
  
  # Return a named list for this class
  return(list(class = model_ovr))
}
stopCluster(cl)
models_ovr = setNames(models_ovr, ovr_classes)
Sys.time()

models_ovr

#for (class in ovr_classes) {
#   ovr_train_data$binary_label = ifelse(train_data$NIBRSDesc == class, 1, 0)
#   
#   models_ovr[[class]] = svm(binary_label ~ .-NIBRSDesc, data = ovr_train_data, 
#                             kernel = "radial", cost = 1, gamma = 0.1)
# }

str(models_ovr)

#evaluating the model
# Predict confidence scores for each class
ovr_scores = data.frame(matrix(NA, nrow = nrow(test_data), ncol = length(ovr_classes)))
colnames(ovr_scores) = ovr_classes

for (class in ovr_classes) {
  ovr_scores[[class]] = predict(models_ovr[[class]], newdata = test_data, decision.values = TRUE)
}

# assign class with highest score 
predictions_ovr = colnames(ovr_scores)[apply(ovr_scores, 1, which.max)]

predictions_ovr_factors = factor(predictions_ovr)
levels(predictions_ovr_factors)
levels(test_data$NIBRSDesc)

confusionMatrix(predictions_ovr_factors, test_data$NIBRSDesc)

ovr_r_conf_matrix = confusionMatrix(as.factor(predictions_ovr_factors), as.factor(test_data$NIBRSDesc))

ovr_r_conf_matrix_df = as.data.frame(r_conf_matrix$table)

ggplot(ovr_r_conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Confusion Matrix Heatmap - OVR Radial Kernel", x = "Predicted Class", y = "Actual Class") +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ovr_svm_roc_curve = multiclass.roc(test_data$NIBRSDesc, as.numeric(predictions_ovr_factors))
ovr_svm_auc_score = auc(ovr_svm_roc_curve)

ovr_svm_auc_score