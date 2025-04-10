getwd()

# 1. Ensure you have a folder "Data" in your working directory that stores the three datasets.
# 2. Ensure you have a folder "PythonDatasets" in your working directory. 
# 3. For reproducability, store the six training and testing data uploaded for each of the three datasets 
#     into the "PythonDatasets" folder for usage in Jupyter Notebook.
# 4. For replicability of the metalearner implementations, run the data preparation code in this script 
#     (Section 1c & 2c) before switching to Jupyter Notebook.

twinsdataX <- read.csv("Data/twin_pairs_X_3years_samesex.csv", header = TRUE, stringsAsFactors = FALSE)
twinsdataY <- read.csv("Data/twin_pairs_Y_3years_samesex.csv", header = TRUE, stringsAsFactors = FALSE)
twinsdataT <- read.csv("Data/twin_pairs_T_3years_samesex.csv", header = TRUE, stringsAsFactors = FALSE)

############################################
############## Data Cleaning ###############
############################################
#install.packages("dplyr")
library(dplyr)
treatment <- twinsdataT %>%
  mutate(
    dbirwt_0 = ifelse(dbirwt_0 < 2500, 1, 0),
    dbirwt_1 = ifelse(dbirwt_1 < 2500, 1, 0))

covariates <- twinsdataX %>%
  select(-c(Unnamed..0,bord_0,bord_1,infant_id_0,infant_id_1))

data <- treatment %>%
  left_join(twinsdataY, by = "X") %>%
  left_join(covariates, by = "X") %>%
  na.omit() # Remove entries that have blanks in any information

W <- data[,2:3]
Y <- data[,4:5]
X <- data[,6:53]

W_0 <- W[,1] # twin_0
W_1 <- W[,2] # twin_1
Y_0 <- Y[,1] # mort_0
Y_1 <- Y[,2] # mort_1

####################################################################################################
######################################## 1. Separately: ############################################
#################################### for twin_0 and twin_1 #########################################
####################################################################################################

##########################################################
########### 1a. Implementing Causal Forests ##############
##########################################################
#install.packages("grf")
#install.packages("randomForest")
library(grf)
library(randomForest)

# CATE estimates using a causal forest:
## twin_0:
cf_0 <- causal_forest(X=X, Y=Y_0, W=W_0, seed=4231) 
cf_0$predictions
## twin_1:
cf_1 <- causal_forest(X=X, Y=Y_1, W=W_1, seed=4231) 
cf_1$predictions

# Covariates Predictive Powers:
## twin_0:
df_0 <- data.frame(CATE=cf_0$predictions,X)
rfcate_0 <- randomForest(CATE~.,data=df_0)
importance(rfcate_0)
importance_0 <- data.frame(Feature = rownames(importance(rfcate_0)), Importance = importance(rfcate_0)[, 1])
importance_0 <- importance_0 %>%
  arrange(desc(Importance))
importance_0$Importance <- round(importance_0$Importance, digits = 4)

## twin_1:
df_1 <- data.frame(CATE=cf_1$predictions,X)
rfcate_1 <- randomForest(CATE~.,data=df_1)
importance(rfcate_1)
importance_1 <- data.frame(Feature = rownames(importance(rfcate_1)), Importance = importance(rfcate_1)[, 1])
importance_1 <- importance_1 %>%
  arrange(desc(Importance))
importance_1$Importance <- round(importance_1$Importance, digits = 4)

# Examining treatment effect across top 5 important groups:
## twin_0:
best_linear_projection(cf_0,data$gestat10)
best_linear_projection(cf_0,data$mplbir)
best_linear_projection(cf_0,data$stoccfipb)
best_linear_projection(cf_0,data$brstate)
best_linear_projection(cf_0,data$birmon)

## twin_1:
best_linear_projection(cf_1,data$gestat10)
best_linear_projection(cf_1,data$birmon)
best_linear_projection(cf_1,data$mplbir) 
best_linear_projection(cf_1,data$brstate)
best_linear_projection(cf_1,data$stoccfipb)

###########################################################
############# 1b. Implementing Policy Tree ################
###########################################################
library(grf)
#install.packages("Matching")
#install.packages("policytree")
#install.packages("DiagrammeR")
library(Matching)
library(policytree)
library(DiagrammeR)

## twin_0:
# Top 5 covariates for twin_0
dr_scores_0 <- double_robust_scores(cf_0)
Xr_0 <- X[,c("gestat10","mplbir","stoccfipb","brstate","birmon")] # Top 5 covariates for twin_0
tree_0 <- policy_tree(Xr_0, dr_scores_0, depth=2) 
plot(tree_0)
tree_0_ <- hybrid_policy_tree(Xr_0, dr_scores_0, depth = 3) # Depth = 3 and higher require hybrid_policy_tree
plot(tree_0_) 

# Top 10 covariates for twin_0
Xr_0 <- X[,c("gestat10","mplbir","stoccfipb","brstate","birmon","dfageq","mplbir_reg","meduc6",
             "dtotord_min", "dlivord_min")] 
tree_0 <- policy_tree(Xr_0, dr_scores_0, depth=2) 
plot(tree_0)
tree_0_ <- hybrid_policy_tree(Xr_0, dr_scores_0, depth = 3) 
plot(tree_0_) 

## twin_1:
# Top 5 covariates for twin_1
dr_scores_1 <- double_robust_scores(cf_1)
Xr_1 <- X[,c("gestat10","mplbir","stoccfipb","brstate","birmon")] 
tree_1 <- policy_tree(Xr_1, dr_scores_1, depth=2) 
plot(tree_1)
tree_1_ <- hybrid_policy_tree(Xr_1, dr_scores_1, depth = 3)
plot(tree_1_)

# Top 10 covariates for twin_1
Xr_1 <- X[,c("gestat10","mplbir","stoccfipb","brstate","birmon","dfageq","nprevistq","mager8",
             "mplbir_reg","dtotord_min")] 
tree_1 <- policy_tree(Xr_1, dr_scores_1, depth=2) 
plot(tree_1)
tree_1_ <- hybrid_policy_tree(Xr_1, dr_scores_1, depth = 3)
plot(tree_1_)

#############################################################
############# 1c. Implementing Metalearners #################
#############################################################
library(dplyr)
library(grf)
library(ggplot2)
library(ranger)  # For random forests
library(caret)

# Prepare the dataset to be used in Matheus Facure Jupyter Notebook

## twin_0:
data_0 <- cbind(X, W_0, Y_0)
# Split into a 80-20 train-test set
data_split_0 <- createDataPartition(data_0$Y_0, p = 0.8, list = FALSE)
train_data_0 <- data_0[data_split_0, ]
test_data_0 <- data_0[-data_split_0, ]
write.csv(train_data_0,"PythonDatasets/train_data_0.csv", row.names = FALSE)
write.csv(test_data_0,"PythonDatasets/test_data_0.csv", row.names = FALSE)

## twin_1:
data_1 <- cbind(X, W_1, Y_1)
# Split into a 80-20 train-test set
data_split_1 <- createDataPartition(data_1$Y_1, p = 0.8, list = FALSE)
train_data_1 <- data_1[data_split_1, ]
test_data_1 <- data_1[-data_split_1, ]
write.csv(train_data_1,"PythonDatasets/train_data_1.csv", row.names = FALSE)
write.csv(test_data_1,"PythonDatasets/test_data_1.csv", row.names = FALSE)

#### Top 10 most predictive covariates for metalearners from Jupyter Notebook
#### Note: Results may differ based on training and testing split. To test reproducability, utilise the
####        uploaded csv used for metalearner implementation to retrieve the following covariates. 
####      For replicability, use the new csvs uploaded. 
## twin_0:
# T-Learners: gestat10 , dtotord_min , mplbir , meduc6 , hydra, mplbir_reg, stoccfipb, feduc6, birmon, mager8
# X-Learners: gestat10 , dtotord_min , meduc6 , incervix , hydra, mager8, nprevistq, pre4000, birmon, brstate

## twin_1:
# T-Learners: gestat10 , mpre5, pldel, hydra , incervix, nprevistq, dfageq, brstate, birmon, dtotord_min
# X-Learners: gestat10 , hydra , brstate , nprevistq , birmon, incervix, csex, dfageq, feduc6, mpre5

####################################################################################################
######################################## 2. Together: ##############################################
####################################################################################################
library(dplyr)

# Contingency Table: Showing that most of the infants who died were those that were LBW
table(W_0,Y_0)
table(W_1,Y_1)

# Combining data: If at least one of the twins has 1 (dbirwt_0/dbirwt_1 = 1), assign 1. 
                  #If at least 1 of the twins died (mort_0/mort_1 = 1), assign 1.
W_pooled <- W %>%
  mutate(LBW = ifelse(dbirwt_0 == 1 | dbirwt_1 == 1, 1, 0))
W_pooled <- W_pooled[,3]
Y_pooled <- Y %>%
  mutate(Mortality = ifelse(mort_0 == 1 | mort_1 == 1, 1, 0))
Y_pooled <- Y_pooled[,3]

##########################################################
########### 2a. Implementing Causal Forests ##############
##########################################################
#install.packages("grf")
#install.packages("randomForest")
library(grf)
library(randomForest)

# CATE estimates using a causal forest:
cf_pooled <- causal_forest(X=X, Y=Y_pooled, W=W_pooled, seed=4231) 
cf_pooled$predictions

# Covariates Predictive Powers:
df_pooled <- data.frame(CATE=cf_pooled$predictions,X)
rfcate_pooled <- randomForest(CATE~.,data=df_pooled)
importance(rfcate_pooled)
importance_pooled <- data.frame(Feature = rownames(importance(rfcate_pooled)), Importance = importance(rfcate_pooled)[, 1])
importance_pooled <- importance_pooled %>%
  arrange(desc(Importance))
importance_pooled$Importance <- round(importance_pooled$Importance, digits = 4)

# Examining treatment effect across top 5 important groups:
best_linear_projection(cf_pooled,data$gestat10)
best_linear_projection(cf_pooled,data$mplbir)
best_linear_projection(cf_pooled,data$stoccfipb)
best_linear_projection(cf_pooled,data$dfageq)
best_linear_projection(cf_pooled,data$brstate)

###########################################################
############# 2b. Implementing Policy Tree ################
###########################################################
library(grf)
#install.packages("Matching")
#install.packages("policytree")
#install.packages("DiagrammeR")
library(Matching)
library(policytree)
library(DiagrammeR)

dr_scores_pooled <- double_robust_scores(cf_pooled)

# Top 5 covariates for combined dataset
Xr_pooled <- X[,c("gestat10","mplbir","stoccfipb","brstate","dfageq")] 
tree_pooled <- policy_tree(Xr_pooled, dr_scores_pooled, depth=2) 
plot(tree_pooled)
tree_pooled_ <- hybrid_policy_tree(Xr_pooled, dr_scores_pooled, depth = 3) 
plot(tree_pooled_)

# Top 10 covariates for combined dataset
Xr_pooled <- X[,c("gestat10","mplbir","stoccfipb","brstate","dfageq","birmon","mager8","nprevistq",
                  "meduc6","dlivord_min")]
tree_pooled <- policy_tree(Xr_pooled, dr_scores_pooled, depth=2) 
plot(tree_pooled)
tree_pooled_ <- hybrid_policy_tree(Xr_pooled, dr_scores_pooled, depth = 3) 
plot(tree_pooled_)

#############################################################
############# 2c. Implementing Metalearners #################
#############################################################
library(dplyr)
library(grf)
library(ggplot2)
library(ranger)  # For random forests
library(caret)

# Prepare the dataset to be used in Matheus Facure Jupyter Notebook
data_pooled <- cbind(X, W_pooled, Y_pooled)

# Split into a 80-20 train-test set
data_split <- createDataPartition(data_pooled$Y_pooled, p = 0.8, list = FALSE)
train_data_pooled <- data_pooled[data_split, ]
test_data_pooled <- data_pooled[-data_split, ]
write.csv(train_data_pooled,"PythonDatasets/train_data_pooled.csv", row.names = FALSE)
write.csv(test_data_pooled,"PythonDatasets/test_data_pooled.csv", row.names = FALSE)

y <- "Y_pooled"
w <- "W_pooled"
x <- c("pldel","birattnd","brstate","stoccfipb","mager8","ormoth","mrace","meduc6","dmar","mplbir",
       "mpre5" ,"adequacy","orfath","frace","birmon","gestat10","csex","anemia","cardiac","lung",
       "diabetes","herpes","hydra","hemo","chyper","phyper","eclamp","incervix","pre4000","preterm",
       "renal","rh","uterine","othermr","tobacco","alcohol","cigar6","drink5","crace","data_year",
       "nprevistq","dfageq","feduc6","dlivord_min","dtotord_min","brstate_reg","stoccfipb_reg",
       "mplbir_reg")

#### Top 10 most predictive covariates for metalearners from Jupyter Notebook
# T-Learners: gestat10 , mpre5 , hydra , nprevistq , mager8, dtotord_min, feduc6, dfageq, brstate, frace
# X-Learners: gestat10 , mpre5 , nprevistq , hydra , mplbir, feduc6, ormoth, diabetes, brstate, meduc6

####################################################################################################
############################## 3. Implementing Overall Policy Tree  ################################
####################################################################################################
############### Combined with the findings from separating the twins and metalearners ##############
####################################################################################################
dr_scores_pooled <- double_robust_scores(cf_pooled)

# Union of all 14 covariates for optimal policy
Xr_pooled <- X[,c("gestat10","mplbir","stoccfipb","brstate","dfageq", "mpre5", "hydra", "nprevistq",
                  "mager8", "dtotord_min", "meduc6", "incervix", "birmon", "pldel")] 
tree_pooled <- policy_tree(Xr_pooled, dr_scores_pooled, depth=2) 
plot(tree_pooled)

tree_pooled_ <- hybrid_policy_tree(Xr_pooled, dr_scores_pooled, depth = 3) 
plot(tree_pooled_)
