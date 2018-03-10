library(rvest)
library(stringr)
library(plyr)

##############History of Salaries

rm(Salary_DF)
USA_COlnames <- c('Rank','Name','Team','POS','SALARY','YEARS','Total_Value')
Salary_DF <- data.frame(matrix(ncol = length(USA_COlnames), nrow = 0))
colnames(Salary_DF) <- USA_COlnames

for (x in 2000:2014){
  url <- paste('https://www.usatoday.com/sports/nhl/salaries/',as.character(x),'/player/all/',sep='')
  
  #Reading the HTML code from the website
  
  webpage <- read_html(url)
  rank_data_html <- html_nodes(webpage,'td')
  data <- html_text(rank_data_html)
  
  #Clean the Data
  
  Contract_Data <- as.data.frame(matrix(data, ncol = 7,  byrow = TRUE), stringsAsFactors = FALSE)
  Contract_Data_Clean <- as.data.frame(lapply(Contract_Data, gsub, pattern = "\n", replacement = "", fixed = TRUE))
  Contract_Data_Clean <- as.data.frame(lapply(Contract_Data_Clean, gsub, pattern = "--", replacement = "", fixed = TRUE))
  Contract_Data_Clean <- data.frame(lapply(Contract_Data_Clean, trimws), stringsAsFactors = FALSE)
  
  #DataFrames
  
  colnames(Contract_Data_Clean) <- USA_COlnames
  
  #Join
  
  Salary_DF <- rbind(Salary_DF,Contract_Data_Clean)}

#Get Year 

Salary_DF$Year <- substr(Salary_DF$YEARS,regexpr('\\(',Salary_DF$YEARS) + 1,regexpr('\\)',Salary_DF$YEARS)-1)

#Breakout By Position Group

Salary_DF$Positional_Group <- ifelse(Salary_DF$POS == 'RW' | Salary_DF$POS == 'LW' | Salary_DF$POS == 'C','Forward',ifelse(Salary_DF$POS == 'G','Goalie','Defense'))

#Change Salary to Number

Salary_DF$SALARY <- gsub(',','',Salary_DF$SALARY)
Salary_DF$SALARY <- gsub('\\$ ','',Salary_DF$SALARY)
Salary_DF$SALARY <- as.numeric(as.character(Salary_DF$SALARY))


#Sum Up

ddply(Salary_DF[2:length(colnames(Salary_DF))],.(Year,Positional_Group),transform,sum(SALARY))
Salary_DF[0]


write.csv(Salary_DF, file = "NHL2000to2014Contracts.csv",,row.names=FALSE)


