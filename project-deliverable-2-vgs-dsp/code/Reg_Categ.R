library(psych)
library(car) 
#install.packages("dummies")
library(dummies)
library(tidyverse)

#Reading the data
#vgdata = read.csv("C:\\Users\\PersonalSpace\\Desktop\\MSIS 5223 PDS2\\project\\vg_data_full.csv")
setwd("C:/Users/Aaron/Google Drive/School Stuff/Spring 2020/Data Science Programming II/project-deliverable-2-vgs-dsp/data")
vgdata = read.csv("vg_data_full.csv")
head(vgdata)
summary(vgdata)

#Missing data removal
#complete.cases(vgdata)
#vgdata = na.omit(vgdata)

#Count null values


range(vgdata$units_total)
hist(vgdata$units_total)

bin_interval = seq(15642, 19392497, by = 62305)
bin_interval

table(cut(vgdata$units_total, bin_interval, right = FALSE))


bin_interval = seq(15642, 19392497, by = 12461)
table(cut(vgdata$units_total, bin_interval, right = TRUE))

table(cut(vgdata$units_total, bin_interval, right = FALSE))


bin_interval = seq(15642, 19404958, by = 12461)
table(cut(vgdata$units_total, bin_interval, right = FALSE))

bin_interval = c(15642, 40564, 65486, 90408, 115330, 127791, 152713, 177635, 202557,19404958)
table(cut(vgdata$units_total, bin_interval, right = FALSE))


bin_interval = c(15642, 1261742, 2507842, 3753942, 19404958)
table(cut(vgdata$units_total, bin_interval, right = FALSE))

age_vector = recode(vgdata$units_total, "15642:641692='15642-641692';1261742:1884792='1261742-1884792'; 
                    2507842:3130892='2507842-3130892'; 3753942:4376992='3753942-4376992'; 
                    5000042:19392497='5000042-19392497'")

vgdata$units_total_categ = age_vector
vgdata[, c('units_total', 'units_total_categ')]

#creating dummy variables for units_total
vg_dummy1 = dummy(vgdata$units_total_categ, sep = '_')
colnames(vg_dummy1)

#Converting the newly created dummies into a dataframe object and then placing back into the dataframe
vg_dummy1 = as.data.frame(vg_dummy1)
vgdata = data.frame(vgdata, vg_dummy1)

vgdata

#binning for metascore
range(vgdata$metascore)
hist(vgdata$metascore)

bin_interval2 = seq(39, 97, by = 5)
bin_interval2

table(cut(vgdata$metascore, bin_interval2, right = FALSE))

bin_interval2 = seq(39, 99, by = 2)
bin_interval2

table(cut(vgdata$metascore, bin_interval2, right = FALSE))

bin_interval2 = c(39, 79, 83, 87, 91, 99)
table(cut(vgdata$metascore, bin_interval2, right = FALSE))

metascore_vector = recode(vgdata$metascore, "39:78='39-78';79:82='79-82'; 
                    83:86='83-86'; 87:91='87-91'; 91:97='91-97'")

vgdata$metascore_categ = metascore_vector
vgdata[, c('metascore', 'metascore_categ')]

#dummy variable for metascore
vg_dummy2 = dummy(vgdata$metascore_categ, sep = '_')
colnames(vg_dummy2)

#Converting the newly created dummies into a dataframe object and then placing back into the dataframe
vg_dummy2 = as.data.frame(vg_dummy2)
vgdata = data.frame(vgdata, vg_dummy2)
vgdata

#binning for user_score
range(vgdata$user_score)
hist(vgdata$user_score)

bin_interval3 = seq(1.4, 9.1, by = 5)
bin_interval3

table(cut(vgdata$user_score, bin_interval3, right = FALSE))

bin_interval3 = seq(1.4, 11.1, by = 2)
bin_interval3

table(cut(vgdata$user_score, bin_interval3, right = FALSE))

bin_interval3 = c(1.4, 5.9, 6.9, 7.9, 9.4)
table(cut(vgdata$user_score, bin_interval3, right = FALSE))

user_score_vector = recode(vgdata$user_score, "1.4:5.8='1.4-5.8';5.9:6.8='5.8-6.8'; 
                    6.9:7.8='6.9-7.8'; 7.9:9.4='7.9-9.4'")

vgdata$user_score_categ = user_score_vector
vgdata[, c('user_score', 'user_score_categ')]

#dummy variable for user_score
vg_dummy3 = dummy(vgdata$user_score_categ, sep = '_')
colnames(vg_dummy3)

#Converting the newly created dummies into a dataframe object and then placing back into the dataframe
vg_dummy3 = as.data.frame(vg_dummy3)
vgdata = data.frame(vgdata, vg_dummy3)
vgdata

range(vgdata$title)
vgdata$title = as.factor(vgdata$title)
