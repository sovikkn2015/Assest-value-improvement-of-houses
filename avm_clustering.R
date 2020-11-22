library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

options("scipen"=100, "digits"=4)

# ############################# -------------------------------------------

data_df = read.csv("Redlands Simplified.csv", header = TRUE, sep = ",")
colnames_df <- colnames(data_df)

targetAPN <- data_df[data_df$apn == "175111130000", ]
data_df2<- data_df[-c(data_df$apn == "175111130000"), ]
data_df<- data_df2

data_df<- data_df[c(data_df$situs_zip_code > 1000000), ]


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

select_df_cluster <- select_df[c("situs_zip_code","avm_value_amount")]
unique_zip <- unique(select_df_cluster["situs_zip_code"])

#library(plyr)
#select_df_cluster["situs_zip_code"] <- round_any(select_df_cluster["situs_zip_code"], 100)
select_df_cluster["situs_zip_code_round"] <- round(select_df_cluster["situs_zip_code"]/10) * 10
unique_zip_round <- unique(select_df_cluster["situs_zip_code_round"])

select_df_cluster["situs_zip_code_round_100"] <- round(select_df_cluster["situs_zip_code"]/100) * 100

#write.csv(select_df_cluster,"select_df_cluster2.csv", row.names = FALSE)

select_df_cluster2 = select_df_cluster[c("situs_zip_code_round_100","avm_value_amount")]

select_df_group1 <- aggregate(select_df_cluster2[,2], list(select_df_cluster2$situs_zip_code_round_100), mean)
select_df_group1$mean <- select_df_group1[,2]
select_df_group1 <- select_df_group1[-c(2)]
select_df_group2 <- aggregate(select_df_cluster2[,2], list(select_df_cluster2$situs_zip_code_round_100), min)
select_df_group2$min <- select_df_group2[,2]
select_df_group2 <- select_df_group2[-c(2)]
select_df_group3 <- aggregate(select_df_cluster2[,2], list(select_df_cluster2$situs_zip_code_round_100), max)
select_df_group3$max <- select_df_group3[,2]
select_df_group3 <- select_df_group3[-c(2)]

select_df_groupall <- Reduce(merge, list(select_df_group1, select_df_group2, select_df_group3))


#write.csv(select_df_groupall,"select_df_groupall.csv", row.names = FALSE)

select_df_clusterselect <- select_df_groupall[,1:2]

# ############################# -------------------------------------------

# Cluster by Round 10

select_df_cluster2 = select_df_cluster[c("situs_zip_code_round","avm_value_amount")]

select_df_group1 <- aggregate(select_df_cluster2[,2], list(select_df_cluster2$situs_zip_code_round), mean)
select_df_group1$mean <- select_df_group1[,2]
select_df_group1 <- select_df_group1[-c(2)]
select_df_group2 <- aggregate(select_df_cluster2[,2], list(select_df_cluster2$situs_zip_code_round), min)
select_df_group2$min <- select_df_group2[,2]
select_df_group2 <- select_df_group2[-c(2)]
select_df_group3 <- aggregate(select_df_cluster2[,2], list(select_df_cluster2$situs_zip_code_round), max)
select_df_group3$max <- select_df_group3[,2]
select_df_group3 <- select_df_group3[-c(2)]

select_df_groupall <- Reduce(merge, list(select_df_group1, select_df_group2, select_df_group3))


#write.csv(select_df_groupall,"select_df_groupall_round10.csv", row.names = FALSE)

select_df_clusterselect <- select_df_groupall[,1:2]

# ############################# -------------------------------------------

# BY KMEANS

# load required packages
library(factoextra)
library(NbClust)

# Elbow method
# fviz_nbclust(select_df_cluster2, kmeans, method = "wss") +
#   geom_vline(xintercept = 4, linetype = 2) + # add line for better visualisation
#   labs(subtitle = "Elbow method") # add subtitle


# Silhouette method
fviz_nbclust(select_df_clusterselect, kmeans, method = "silhouette") +
  labs(subtitle = "Silhouette method")

# distance <- get_dist(select_df_cluster)
# fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))



# Compute k-means with k = 3
set.seed(123)
res.km <- kmeans(select_df_clusterselect, 2, nstart = 5)
# K-means clusters showing the group of each individuals
res.km$cluster

select_df_groupall$cluster <- res.km$cluster

fviz_cluster(res.km, data = select_df_clusterselect,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

write.csv(select_df_groupall,"select_df_groupall_cluster.csv", row.names = FALSE)


# ############################# -------------------------------------------

# BY PCA

library(ggpubr)

# Dimension reduction using PCA
res.pca <- prcomp(select_df_clusterselect,  scale = TRUE)
# Coordinates of individuals
ind.coord <- as.data.frame(get_pca_ind(res.pca)$coord)
# Add clusters obtained using the K-means algorithm
ind.coord$cluster <- factor(res.km$cluster)
# Add Species groups from the original data sett
#ind.coord$Species <- df$Species
# Data inspection
head(ind.coord)

# Percentage of variance explained by dimensions
eigenvalue <- round(get_eigenvalue(res.pca), 1)
variance.percent <- eigenvalue$variance.percent
head(eigenvalue)

ggscatter(
  ind.coord, x = "Dim.1", y = "Dim.2", 
  color = "cluster", palette = "npg", ellipse = TRUE, ellipse.type = "convex",
  #shape = "Species", 
  size = 1.5,  legend = "right", ggtheme = theme_bw(),
  xlab = paste0("Dim 1 (", variance.percent[1], "% )" ),
  ylab = paste0("Dim 2 (", variance.percent[2], "% )" )
) +
  stat_mean(aes(color = cluster), size = 4)
