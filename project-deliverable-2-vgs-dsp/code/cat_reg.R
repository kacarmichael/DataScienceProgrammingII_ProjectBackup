library(tidyverse)
library(psych)
library(car) 
library(dummies)
library(data.table)

#Reading the vgdata
setwd("C:/Users/Aaron/Google Drive/School Stuff/Spring 2020/vgdata Science Programming II/project-deliverable-2-vgs-dsp/data")
vgdata = read.csv("vg_data_full.csv", stringsAsFactors = FALSE)
head(vgdata)
summary(vgdata)


#Fixing faulty target values
vgdata[1145,]$units_total <- 54137
vgdata[1146,]$units_total <- 3918368
vgdata[1147,]$units_total <- 48234
vgdata[1148,]$units_total <- 74488
vgdata[1149,]$units_total <- 17799238
vgdata[1150,]$units_total <- 42659
vgdata[1151,]$units_total <- 12586766
vgdata[1152,]$units_total <- 116383
vgdata[1153,]$units_total <- 56730
vgdata[1154,]$units_total <- 478351
vgdata[1155,]$units_total <- 1351858
vgdata[1156,]$units_total <- 1820330
vgdata[1157,]$units_total <- 3748058
vgdata[1158,]$units_total <- 1663760
vgdata[1159,]$units_total <- 23876
vgdata[1160,]$units_total <- 1020739
vgdata[1161,]$units_total <- 3997120
vgdata[1162,]$units_total <- 3815161


#Categorizing player number
player_vals <- unique(vgdata$numPlayers)
single_player <- c("No Online Multiplayer")
co_op <- c("Up to 6", "Up to 4", "2", "Up to 5")
multi_player <- player_vals[(!player_vals %in% co_op) && (!player_vals %in% single_player) && (!player_vals == "NA")]

vgdata$player <- ""       
for (i in 1:length(vgdata$numPlayers)) {
  if(vgdata[i,]$numPlayers %in% single_player) {
    vgdata[i,]$player = "single"
  }
  else if(vgdata[i,]$numPlayers %in% co_op) {
    vgdata[i,]$player = "co_op"
  }
  else if(vgdata[i,]$numPlayers %in% multi_player) {
    vgdata[i,]$player = "multi"
  }
}

#Ordered based on player number
player_vector = recode(vgdata$player, 'single' = '0', 'co_op' = '1', 'multi' = '2')



#PSVR Ratings
vgdata[vgdata$title == "Batman: Arkham VR",]$esrb = "M"
vgdata[vgdata$title == "PlayStation VR Worlds",]$esrb = "M"
vgdata[vgdata$title == "Robinson: The Journey",]$esrb = "E"
vgdata[vgdata$title == "Until Dawn: Rush of Blood",]$esrb = "M"

#Missing Ratings: fixing 40% of missing ratings
vgdata[vgdata$title == "Assassin's Creed: Unity",]$esrb = "M"
vgdata[vgdata$title == "FIFA 16",]$esrb = "E"
vgdata[vgdata$title == "FIFA 18",]$esrb = "E"
vgdata[vgdata$title == "Battlefield: Hardline",]$esrb = "M"
vgdata[vgdata$title == "FIFA 17",]$esrb = "E"
vgdata[vgdata$title == "Destiny: The Taken King",]$esrb = "M"
vgdata[vgdata$title == "NBA 2K17",]$esrb = "E"
vgdata[vgdata$title == "MLB 16: The Show",]$esrb = "E"
vgdata[vgdata$title == "Wolfenstein: The New Order",]$esrb = "M"
vgdata[vgdata$title == "FIFA 15",]$esrb = "E"


#Ordered based on maturity level

esrb_vector = recode(vgdata$esrb, 'E' = '0', 'E10+' = '1', 'T' = '2', 'M' = '3')

#Average of all MGScores
vgdata$mg1_mgscore <- as.numeric(vgdata$mg1_mgscore)
vgdata$mg2_mgscore <- as.numeric(vgdata$mg2_mgscore)
vgdata$mg3_mgscore <- as.numeric(vgdata$mg3_mgscore)
vgdata$mg4_mgscore <- as.numeric(vgdata$mg4_mgscore)
vgdata$mg5_mgscore <- as.numeric(vgdata$mg5_mgscore)
vgdata$mg6_mgscore <- as.numeric(vgdata$mg6_mgscore)
vgdata$mg7_mgscore <- as.numeric(vgdata$mg7_mgscore)
vgdata$mg8_mgscore <- as.numeric(vgdata$mg8_mgscore)
vgdata$mg9_mgscore <- as.numeric(vgdata$mg9_mgscore)
vgdata$mg10_mgscore <- as.numeric(vgdata$mg10_mgscore)

vgdata$mgAverage <- rowMeans(vgdata[c('mg1_mgscore', 'mg2_mgscore', 'mg3_mgscore', 'mg4_mgscore', 'mg5_mgscore',
                                  'mg6_mgscore', 'mg7_mgscore', 'mg8_mgscore', 'mg9_mgscore', 'mg10_mgscore')], na.rm = TRUE)

#Console dummies
console_dummy = as.data.frame(dummy(vgdata$console, sep="_"))
colnames(console_dummy) <- c("console_PS4", "console_Switch", "console_XBox1")

#Genre dummies
genre_dummy = as.data.frame(dummy(vgdata$genre, sep="_"))
colnames(genre_dummy) <- c("genre_Action", "genre_ActionAdventure", "genre_Adventure",
                           "genre_Fighting", "genre_Misc", "genre_Music", "genre_Party",
                           "genre_Platform", "genre_Puzzle", "genre_Racing", "genre_RPG",
                           "genre_Sandbox", "genre_Shooter", "genre_Sim", "genre_Sports",
                           "genre_Strategy")

#Combine Action, Adventure, Action-Adventure
genre_dummy$genre_Action_Adventure <- genre_dummy$genre_Action + genre_dummy$genre_ActionAdventure + genre_dummy$genre_Adventure
genre_dummy <- genre_dummy[!colnames(genre_dummy) %in% c("genre_Action", "genre_ActionAdventure", "genre_Adventure")]

#Combine Misc and all genres with less than 100 records (Party, Strategy, Simulation, Sandbox, Puzzle, Music)
genre_dummy$genre_Misc <- genre_dummy$genre_Misc + genre_dummy$genre_Party + genre_dummy$genre_Strategy + genre_dummy$genre_Sim + 
  genre_dummy$genre_Sandbox + genre_dummy$genre_Puzzle + genre_dummy$genre_Music

genre_dummy <- genre_dummy[!colnames(genre_dummy) %in% c("genre_Action", "genre_ActionAdventure", "genre_Adventure",
                                                         "genre_Party", "genre_Strategy", "genre_Sim", "genre_Sandbox", "genre_Puzzle", "genre_Music")]


vgdata <- data.frame(vgdata, console_dummy, genre_dummy, player_vector, esrb_vector)

full_data <- vgdata %>%
  mutate(ign_rating = as.numeric(ign_rating)) %>%
  mutate(ign_x10 = ign_rating*10) %>%
  select(title, console_PS4, console_XBox1, console_Switch, publisher,
         genre_Fighting, genre_Misc, genre_Platform, genre_Racing,
         genre_RPG, genre_Shooter, genre_Sports, genre_Action_Adventure,
         units_total, player_vector, esrb_vector, metascore,
         user_score_x10, igdb_member_rating, 
         igdb_critic_rating, ign_x10, mgAverage) 

sales_data <- full_data %>%
  group_by(title) %>%
  summarise(sales = max(units_total))


info_data <- full_data %>%
  select(-c(publisher, units_total)) %>%
  group_by(title) %>%
  distinct(title, .keep_all = TRUE)

reg_data <- sales_data %>%
  left_join(y=info_data, by="title")


#Count null values
sapply(reg_data, function(x) sum(is.na(x)))
  
reg_data$metascore <- as.numeric(reg_data$metascore)
reg_data$user_score_x10 <- as.numeric(reg_data$user_score_x10)
reg_data$igdb_member_rating <- as.numeric(reg_data$igdb_member_rating)
reg_data$igdb_critic_rating <- as.numeric(reg_data$igdb_critic_rating)
#Replacing null scores with average by row

reg_data <- reg_data %>% 
  rowwise() %>%
  mutate(avg_score = mean(c(metascore, user_score_x10,
                          igdb_member_rating,
                          igdb_critic_rating,
                          ign_x10,
                          mgAverage), na.rm = TRUE))

for (row in c(1:nrow(reg_data))) {
  for (col in c(16:21)) {
    if (is.na(reg_data[row, col])) {
      reg_data[row, col] = reg_data[row,]$avg_score
    }
  }
}

reg_data <- subset(reg_data, select = c(-avg_score))

#Replacing Null ESRB values
summary(reg_data$esrb_vector)

#M rating (3) is most occuring rating
for (row in c(1:nrow(reg_data))) {
  if(is.na(reg_data[row,]$esrb_vector)) {
    reg_data[row,]$esrb_vector = 3
  }
}

sapply(reg_data, function(x) sum(is.na(x)))

#Binning units_total

range(reg_data$sales)
hist(reg_data$sales)

bin_interval = seq(15642, 19392497, by = 62305)
bin_interval

table(cut(reg_data$sales, bin_interval, right = FALSE))


bin_interval = seq(15642, 19392497, by = 12461)
table(cut(reg_data$sales, bin_interval, right = TRUE))

table(cut(reg_data$sales, bin_interval, right = FALSE))


bin_interval = seq(15642, 19404958, by = 12461)
table(cut(reg_data$sales, bin_interval, right = FALSE))

bin_interval = c(15642, 40564, 65486, 90408, 115330, 127791, 152713, 177635, 202557,19404958)
table(cut(reg_data$sales, bin_interval, right = FALSE))


bin_interval = c(15642, 1261742, 2507842, 3753942, 19404958)
table(cut(reg_data$sales, bin_interval, right = FALSE))

bin_interval = c(0, 60500, 215000, 1150000, 20000000)
bin_labels = c("I", "II", "III", "IV")
setDT(reg_data)[, salesgroups := cut(sales, breaks = bin_interval, right = FALSE, labels = bin_labels)]
table(cut(reg_data$sales, bin_interval, right = FALSE))

head(reg_data)
str(reg_data)

write.csv(reg_data, "cat_reg_data.csv")
