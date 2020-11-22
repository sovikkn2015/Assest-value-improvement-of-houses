library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
#library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization
library(tidyverse)

options("scipen"=100, "digits"=4)

# ############################# -------------------------------------------


data_df = read.csv("Redlands_selected_apn.csv", header = TRUE, sep = ",")
colnames_df <- colnames(data_df)

#targetAPN <- data_df[data_df$apn == "175111130000", ]
#data_df2<- data_df[-c(data_df$apn == "175111130000"), ]
#data_df<- data_df2

selectvars <- c("number_of_buildings","situs_zip_code","assessed_land_value","land_square_footage",
                "year_built","bedrooms_all_buildings","total_rooms_all_buildings","total_bath_rooms_calculated_all_buildings",
                "total_baths_primary_building","full_baths_all_buildings","half_baths_all_buildings","pool_indicator",
                "stories_number","universal_building_square_feet","building_square_feet","living_square_feet_all_buildings",
                "ground_floor_square_feet","basement_square_feet","garage_parking_square_feet","avm_value_amount")
select_df <- data_df[selectvars]

select_df[is.na(select_df)] <- 0
#sapply(select_df, class)

select_df$pool_indicator <- as.character(select_df$pool_indicator)
select_df$pool_indicator[select_df$pool_indicator == "N"] <- 0
select_df$pool_indicator[select_df$pool_indicator == "Y"] <- 1
select_df$pool_indicator <- as.integer(select_df$pool_indicator)
select_df[is.na(select_df)] <- 0

# ############################# -------------------------------------------

# 1 Bedroom Outside

bedseq <- seq(0, 500, by=25)
select_df_bed <- data_df

select_df_bed <- rename(select_df_bed,c("bedrooms_all_buildings_old" = "bedrooms_all_buildings",
                                        "total_rooms_all_buildings_old" = "total_rooms_all_buildings",
                                        "universal_building_square_feet_old" = "universal_building_square_feet",
                                        "building_square_feet_old" = "building_square_feet"))

select_df_bed <- as.data.frame(lapply(select_df_bed, rep, length(bedseq)))
select_df_bed$bedrooms_all_buildings[2:length(bedseq)] <- select_df_bed$bedrooms_all_buildings_old[2:length(bedseq)]+1
select_df_bed$total_rooms_all_buildings[2:length(bedseq)] <- select_df_bed$total_rooms_all_buildings_old[2:length(bedseq)]+1
select_df_bed$new_bedroom_area <- bedseq
select_df_bed$universal_building_square_feet <- select_df_bed$new_bedroom_area + select_df_bed$universal_building_square_feet_old
select_df_bed$building_square_feet <- select_df_bed$new_bedroom_area + select_df_bed$building_square_feet_old
select_df_bed$building_to_land_sq_footage_ratio_pc <- select_df_bed$building_square_feet*100/select_df_bed$land_square_footage
select_df_bed$Cost_to_build_new <- select_df_bed$new_bedroom_area*180


select_df_bed_pred <- select_df_bed[selectvars]
avm_model <- readRDS("avm_gbm_set1.rds")
predgbm_bed <- predict(avm_model, n.trees = avm_model$n.trees, select_df_bed_pred)

select_df_bed["pred_avm"] <- predgbm_bed

#write.csv(predgbm,"predgbm_1_bathroom_1_bathroom_outside.csv", row.names = FALSE)

write.csv(select_df_bed,"predgbm_bed_set1.csv", row.names = FALSE)


# ############################# -------------------------------------------

# 1 Bedroom Outside

bathseq <- seq(0, 200, by=10)
select_df_bath <- data_df

select_df_bath <- rename(select_df_bath,c("total_bath_rooms_calculated_all_buildings_old" = "total_bath_rooms_calculated_all_buildings",
                                        "total_rooms_all_buildings_old" = "total_rooms_all_buildings",
                                        "universal_building_square_feet_old" = "universal_building_square_feet",
                                        "building_square_feet_old" = "building_square_feet"))

select_df_bath <- as.data.frame(lapply(select_df_bath, rep, length(bathseq)))
select_df_bath$total_bath_rooms_calculated_all_buildings[2:length(bathseq)] <- select_df_bath$total_bath_rooms_calculated_all_buildings[2:length(bathseq)]+1
select_df_bath$total_rooms_all_buildings[2:length(bathseq)] <- select_df_bath$total_rooms_all_buildings_old[2:length(bathseq)]+1
select_df_bath$new_bathroom_area <- bathseq
select_df_bath$universal_building_square_feet <- select_df_bath$new_bathroom_area + select_df_bath$universal_building_square_feet_old
select_df_bath$building_square_feet <- select_df_bath$new_bathroom_area + select_df_bath$building_square_feet_old
select_df_bath$building_to_land_sq_footage_ratio_pc <- select_df_bath$building_square_feet*100/select_df_bath$land_square_footage
select_df_bath$Cost_to_build_new <- select_df_bath$new_bathroom_area*180

select_df_bath_pred <- select_df_bath[selectvars]
avm_model <- readRDS("avm_gbm_set1.rds")
predgbm_bath <- predict(avm_model, n.trees = avm_model$n.trees, select_df_bath_pred)

select_df_bath["pred_avm"] <- predgbm_bath

write.csv(select_df_bath,"predgbm_bath_set1.csv", row.names = FALSE)



# ############################# -------------------------------------------

# 1 Bedroom + 1 Bathroom

bedbathseq <- expand.grid(bedseq,bathseq)
colnames(bedbathseq) <- c("bed", "bath")
bedbathseq[1,] <- c(9999, 9999)
bedbathseq[bedbathseq==0] <- NA
bedbathseq<-bedbathseq[complete.cases(bedbathseq),]
bedbathseq[1,] <- c(0,0)
bedbathseq <- bedbathseq[order(bedbathseq$bed, decreasing = FALSE), ]


select_df_bedbath <- data_df

select_df_bedbath <- rename(select_df_bedbath,c("bedrooms_all_buildings_old" = "bedrooms_all_buildings",
                                          "total_bath_rooms_calculated_all_buildings_old" = "total_bath_rooms_calculated_all_buildings",
                                          "total_rooms_all_buildings_old" = "total_rooms_all_buildings",
                                          "universal_building_square_feet_old" = "universal_building_square_feet",
                                          "building_square_feet_old" = "building_square_feet"))

select_df_bedbath <- as.data.frame(lapply(select_df_bedbath, rep, nrow(bedbathseq)))
select_df_bedbath$bedrooms_all_buildings[2:length(bedbathseq)] <- select_df_bedbath$bedrooms_all_buildings_old[2:length(bedbathseq)]+1
select_df_bedbath$total_bath_rooms_calculated_all_buildings[2:length(bedbathseq)] <- select_df_bedbath$total_bath_rooms_calculated_all_buildings[2:length(bedbathseq)]+1
select_df_bedbath$total_rooms_all_buildings[2:length(bedbathseq)] <- select_df_bedbath$total_rooms_all_buildings_old[2:length(bedbathseq)]+2
x <- bedbathseq["bed"]+bedbathseq["bath"]
colnames(x) <- c("new_bedbathroom_area")
select_df_bedbath <- cbind(select_df_bedbath, x)
#select_df_bedbath$new_bedbathroom_area <- bedbathseq["bed"]+bedbathseq["bath"]
a <- as.data.frame(select_df_bedbath$new_bedbathroom_area + select_df_bedbath$universal_building_square_feet_old)
colnames(a) <- c("universal_building_square_feet")
#select_df_bedbath$universal_building_square_feet <- a
select_df_bedbath <- cbind(select_df_bedbath, a)
b <- as.data.frame(select_df_bedbath$new_bedbathroom_area + select_df_bedbath$building_square_feet_old)
colnames(b) <- c("building_square_feet")
#select_df_bedbath$building_square_feet <- b
select_df_bedbath <- cbind(select_df_bedbath, b)
select_df_bedbath$building_to_land_sq_footage_ratio_pc <- select_df_bedbath$building_square_feet*100/select_df_bedbath$land_square_footage
select_df_bedbath$Cost_to_build_new <- select_df_bedbath$new_bedbathroom_area*180


select_df_bedbath_pred <- select_df_bedbath[selectvars]
avm_model <- readRDS("avm_gbm_set1.rds")
predgbm_bedbath <- predict(avm_model, n.trees = avm_model$n.trees, select_df_bedbath_pred)

select_df_bedbath["pred_avm"] <- predgbm_bedbath

write.csv(select_df_bedbath,"predgbm_bedbath_set1.csv", row.names = FALSE)


# ############################# -------------------------------------------



# avm_model <- readRDS("avm_gbm.rds")
# predgbm <- predict(avm_model, n.trees = avm_model$n.trees, select_df)
# 
# write.csv(predgbm,"predgbm_1_bathroom_1_bathroom_outside.csv", row.names = FALSE)

