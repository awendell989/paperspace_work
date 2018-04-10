#Avery FLoads FIles
library(httr)
library(jsonlite)

#Create Functions

NHLAPI_TO_Char <- function(x) {
  raw.result <- GET(url = x)
  this.raw.content <- rawToChar(raw.result$content)
  transform <- fromJSON(this.raw.content) 
  transform_df <- as.data.frame(transform[2])
  return(transform_df)
}

#Raw URL

api_url <- 'http://statsapi.web.nhl.com/api/v1/'

#Grab All the Teams

teams_add <- 'teams'
teams_api_url <- paste(api_url,teams_add,sep ="")
teams_df <- NHLAPI_TO_Char(teams_api_url)
 
#Identify the Target Team

Target_ID <- teams_df$teams.id[grep('Islanders',teams_df$teams.name)]

#Target Team

Target_Roster_URL <- paste(teams_api_url,'/',as.character(Islanders_ID),'/roster',sep="")
Target_Roster <- NHLAPI_TO_Char(Target_Roster_URL)

#Scheudle

Target_Schedule_URL <- 'https://statsapi.web.nhl.com/api/v1/schedule?startDate=2016-10-01&endDate=2018-04-07&teamId=2' # https://github.com/dword4/nhlapi#schedule
raw.result <- GET(url = Target_Schedule_URL)
this.raw.content <- rawToChar(raw.result$content)
transform <- fromJSON(this.raw.content)
schedule_raw <- transform$dates$games
avery <- schedule_raw[1]

rm(schedule_data_frame)

for (x in 1:length(schedule_raw)){
  game <- as.data.frame(t(unlist(schedule_raw[x])))
  if (exists('schedule_data_frame')){
    rbind(schedule_data_frame,game)
  }
  else {schedule_data_frame <- game}}
  
  
  

  print(as.data.frame(t(unlist(schedule_raw[x]))))
  
}
  


as.data.frame(unlist(avery))




['gamePk']
#Schedule <- as.data.frame(

matrix(unlist(transform$dates$games), nrow= 29)

length()

transform_df <- as.data.frame(transform$dates$games)
                              
                              )
d





player_url <- 'http://statsapi.web.nhl.com/api/v1/people/8479318'


raw.result <- GET(url = player_url)
this.raw.content <- rawToChar(raw.result$content)
transform <- fromJSON(this.raw.content)

#Playe DF

Player_DF <- as.data.frame(transform$people)

this.content.df <- do.call(what = "rbind",args = lapply(transform$people, as.data.frame))

raw.result$cookies
raw.result$all_headers
raw.result$headers
raw.result$status_code
raw.result$url