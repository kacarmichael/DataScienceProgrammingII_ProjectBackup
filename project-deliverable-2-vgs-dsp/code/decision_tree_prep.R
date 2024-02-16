library(tidyverse)
library(dummies)

setwd("C:/Users/Aaron/Google Drive/School Stuff/Spring 2020/Data Science Programming II/project-deliverable-2-vgs-dsp/data")

data <- read.csv("vg_data_full.csv", stringsAsFactors = FALSE)

#Fixing an anomaly caused during the initial scrape of vgchartz

#Faulty chart: March 24, 2018
#Indices: 1145-1162

data[1145,]$units_total <- 54137
data[1146,]$units_total <- 3918368
data[1147,]$units_total <- 48234
data[1148,]$units_total <- 74488
data[1149,]$units_total <- 17799238
data[1150,]$units_total <- 42659
data[1151,]$units_total <- 12586766
data[1152,]$units_total <- 116383
data[1153,]$units_total <- 56730
data[1154,]$units_total <- 478351
data[1155,]$units_total <- 1351858
data[1156,]$units_total <- 1820330
data[1157,]$units_total <- 3748058
data[1158,]$units_total <- 1663760
data[1159,]$units_total <- 23876
data[1160,]$units_total <- 1020739
data[1161,]$units_total <- 3997120
data[1162,]$units_total <- 3815161

  


#TODO: 





player_vals <- unique(data$numPlayers)
single_player <- c("No Online Multiplayer")
co_op <- c("Up to 6", "Up to 4", "2", "Up to 5")
multi_player <- player_vals[(!player_vals %in% co_op) && (!player_vals %in% single_player) && (!player_vals == "NA")]

data$player <- ""       
for (i in 1:length(data$numPlayers)) {
  if(data[i,]$numPlayers %in% single_player) {
    data[i,]$player = "single"
  }
  else if(data[i,]$numPlayers %in% co_op) {
    data[i,]$player = "co_op"
  }
  else if(data[i,]$numPlayers %in% multi_player) {
    data[i,]$player = "multi"
  }
}

#Ordered based on player number
player_vector = recode(data$player, "'single' = '0'; 'co_op' = '1'; 'multi' = '2'")


#Categorize ESRB Rating

#Fix "Playstation VR"and missing Ratings
VR_Ratings <- data %>%
  select(title, esrb) %>%
  filter(is.na(esrb)) %>%
  group_by(title) %>%
  tally() %>%
  arrange(-n) %>%
  mutate(percentage = (n/sum(n))*100) %>%
  mutate(pareto = cumsum(percentage))

#PSVR Ratings
data[data$title == "Batman: Arkham VR",]$esrb = "M"
data[data$title == "PlayStation VR Worlds",]$esrb = "M"
data[data$title == "Robinson: The Journey",]$esrb = "E"
data[data$title == "Until Dawn: Rush of Blood",]$esrb = "M"

#Missing Ratings: fixing 40% of missing ratings
data[data$title == "Assassin's Creed: Unity",]$esrb = "M"
data[data$title == "FIFA 16",]$esrb = "E"
data[data$title == "FIFA 18",]$esrb = "E"
data[data$title == "Battlefield: Hardline",]$esrb = "M"
data[data$title == "FIFA 17",]$esrb = "E"
data[data$title == "Destiny: The Taken King",]$esrb = "M"
data[data$title == "NBA 2K17",]$esrb = "E"
data[data$title == "MLB 16: The Show",]$esrb = "E"
data[data$title == "Wolfenstein: The New Order",]$esrb = "M"
data[data$title == "FIFA 15",]$esrb = "E"

#Ordered based on maturity level
esrb_vector = recode(data$esrb, "'E' = '0'; 'E10+' = '1'; 'T' = '2'; 'M' = '3'")

#Average of all MGScores
data$mg1_mgscore <- as.numeric(data$mg1_mgscore)
data$mg2_mgscore <- as.numeric(data$mg2_mgscore)
data$mg3_mgscore <- as.numeric(data$mg3_mgscore)
data$mg4_mgscore <- as.numeric(data$mg4_mgscore)
data$mg5_mgscore <- as.numeric(data$mg5_mgscore)
data$mg6_mgscore <- as.numeric(data$mg6_mgscore)
data$mg7_mgscore <- as.numeric(data$mg7_mgscore)
data$mg8_mgscore <- as.numeric(data$mg8_mgscore)
data$mg9_mgscore <- as.numeric(data$mg9_mgscore)
data$mg10_mgscore <- as.numeric(data$mg10_mgscore)

data$mgAverage <- rowMeans(data[c('mg1_mgscore', 'mg2_mgscore', 'mg3_mgscore', 'mg4_mgscore', 'mg5_mgscore',
                                  'mg6_mgscore', 'mg7_mgscore', 'mg8_mgscore', 'mg9_mgscore', 'mg10_mgscore')], na.rm = TRUE)

#Console dummies
console_dummy = as.data.frame(dummy(data$console, sep="_"))
colnames(console_dummy) <- c("console_PS4", "console_Switch", "console_XBox1")

#Genre dummies
genre_dummy = as.data.frame(dummy(data$genre, sep="_"))
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


genre_dist <- data %>%
  select(genre, title) %>%
  group_by(genre) %>%
  tally() %>%
  arrange(-n)

data <- data.frame(data, console_dummy, genre_dummy, player_vector, esrb_vector)

full_data <- data %>%
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

dt_data <- sales_data %>%
  left_join(y=info_data, by="title")

write.csv(dt_data, "dt_data.csv")





