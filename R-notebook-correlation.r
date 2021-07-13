library(arcgisbinding)

arc.check_product()

#install.packages("maxnet")

shades_l <- arc.open('C:/Internship_work/Model/model_output.gdb/shadesleidingn_r_g_res_3')
df <- arc.select(shades_l)
print(colnames(df))
head(df,2)

#install.packages("psych")
library(psych)
#install.packages("tidyverse")
library(tidyverse)
select_data = df[, c(8:46, 50:51)]

#check length
#install.packages("polycor")
library(polycor)
my_data <- as_tibble(select_data)
my_data
ncol(my_data)
colnames(my_data)



table_c <- cor(my_data)
print(table_c)
round(table_c,2)
#install.packages("corrplot")
#library(corrplot)
#corrplot(table_c, method="circle")


#png(filename="C:/Internship_work/cor_res_1.jpg")
#plot<-corrplot(table_c, method="circle")
#dev.off()
#arc.write('C:/Internship_work/Model/model_input.gdb/table_c',table_c )
#install.packages("gridExtra")   # Install & load gridExtra
library("gridExtra")

#install.packages("writexl")
library(writexl)
df_c <- as.data.frame(table_c)
df_c
write_xlsx(df_c,"C:/Internship_work/Pearson's correlation coefficient.xlsx")


