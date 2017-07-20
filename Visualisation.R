setwd("C:/Users/Shreya/Documents") 
# read the contents of the file 
#... sep : tells R as to how are the data inputs seperated. In case of csv it is comma
#....dec: decimal value representation in the file
#....Header: If the first row of the file contains description of the field set it to True
MyVar <- read.csv(file = "train_v2.csv", header = TRUE, sep = ",", quote = "\"",
                  dec = ".", fill = TRUE, comment.char = "") 
## 
