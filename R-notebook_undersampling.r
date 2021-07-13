library(arcgisbinding)

arc.check_product()

#install.packages("maxnet")

shades_l <- arc.open('C:/Internship_work/Model/model_input.gdb/waterleidingen_schades_r_g')
df <- arc.select(shades_l)
anders_r <- arc.select(shades_l, where_clause="prob_en_anders='Anders' and DATUM_AANL > 1")
head(anders_r, n=3)

#install.packages('boot', dep = TRUE)
#library(boot)
#install.packages("imbalance")
#library(imbalance)
print(nrow(df[df$prob_en_ander == 'Problem',]))
print(nrow(df[df$prob_en_ander == 'Anders',]))

re_sample <- df[sample(1 : nrow(df),278, prob = ifelse(df$prob_en_anders == 'Problem', 0.013, 0.987)), ]
re_sample$prob_en_anders
nrow(re_sample)
length(re_sample[re_sample$prob_en_ander == 'Problem',])
#see the numbers of rows grouped together
nrow(re_sample[re_sample$prob_en_ander == 'Problem',])
length(re_sample[re_sample$prob_en_ander == 'Anders',])

arc.write('C:/Internship_work/Model/model_input.gdb/shadesleidingn_r_g_res',re_sample )


