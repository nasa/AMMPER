
setwd("C:/Users/danie/Desktop/Daniel_Master_Directory/AMMPER/AMMPER_NEW/AMMPER/StatisticsBulk")

if(!require('stats')) {
  install.packages('stats')
  library('stats')
}
library(readr)
library(dendsort)
library(tidyverse)
library(dplyr)
library(pheatmap)
library(grid)
library(iClusterPlus)
library(iCluster2)
library(GenomicRanges)
library(gplots)
library(lattice)
library(maftools)
library(CopyNumberPlots)
library(rstatix)

test <- read_tsv(file="Complex_vs_Basic_RM2ANOVA.tsv")

test <- as.data.frame(test)

test <- subset(test, select = -1 )

colnames(test)[1] <- "time"
test$dose <- paste(test$ROS_model, test$dose, sep="_")
test$dose <- as.factor(test$dose)
test$variable <- as.factor(test$variable)
test$ROS_model <- NULL
rad51 <- test %>% filter(Treatment == "rad51")
rad51$Treatment <- NULL  
WT <- test %>% filter(Treatment == "WT")
WT$Treatment <- NULL

summary<-rad51 %>%
  group_by(dose,time) %>%
  get_summary_stats(value, type = "mean_sd")
data.frame(summary)

View(rad51)

#### PLOTS 
color1 = "#d31e25"
color2 = "#d7a32e"
color3 = "#369e4b"
color4 = "#5db5b7"
color5 = "#31407b"
color6 = "#d1c02b"
color7 = "#8a3f64"
color8 = "#4f2e39"

library(hrbrthemes)

rad51t14 <- rad51 %>% filter(time == "t14")
a <- rad51t14 %>% ggplot( aes(x=value, fill=dose)) +
  geom_histogram( color="#e9ecef", alpha=0.6, position = 'identity') +
  scale_fill_manual(values=c("#d31e25", "#d7a32e", "#369e4b","#5db5b7")) +
  theme_ipsum() +
  labs(fill="")

a
# Shapiro

normality<-rad51 %>%
  group_by(dose,time) %>%
  shapiro_test(value)
data.frame(normality)


rad51 <- rownames_to_column(rad51)
colnames(rad51)[1] <- "id"

# PAIRWISE WILCOXONS >>>>>>>>>>>>>>>>>>>>>>>>>>>
coli = c("t12", "t13", "t14", "t15")
colj = c("Basic_2.5", "Basic_5", "Complex_2.5", "Complex_5")

for (x in coli) {
  
  for (y in colj) {
    rad51_t <- rad51 %>% filter(time == x)
    
    rad51_tb <- rad51_t %>% filter(dose == "Basic_2.5")
    
    rad51_tb2 <- rad51_t %>% filter(dose == y)
    
    res <- wilcox.test(rad51_tb$value, rad51_tb2$value, paired = FALSE)
    print("STARTING TEST FOR >>>>")
    print(x)
    print(y)
    print(res)
    
  }

  
}

WT <- rownames_to_column(WT)
colnames(WT)[1] <- "id"

for (x in coli) {
  
  for (y in colj) {
    WT_t <- WT %>% filter(time == x)
    
    WT_tb <- WT_t %>% filter(dose == "Basic_2.5")
    
    WT_tb2 <- WT_t %>% filter(dose == y)
    
    res <- wilcox.test(WT_tb$value, WT_tb2$value, paired = FALSE)
    print("STARTING TEST FOR >>>>")
    print(x)
    print(y)
    print(res)
    
  }
  
  
}

################# Basic 5

for (x in coli) {
  
  for (y in colj) {
    rad51_t <- rad51 %>% filter(time == x)
    
    rad51_tb <- rad51_t %>% filter(dose == "Complex_2.5")
    
    rad51_tb2 <- rad51_t %>% filter(dose == y)
    
    res <- wilcox.test(rad51_tb$value, rad51_tb2$value, paired = FALSE)
    print("STARTING TEST FOR >>>>")
    print(x)
    print(y)
    print(res)
    
  }
  
  
}

for (x in coli) {
  
  for (y in colj) {
    WT_t <- WT %>% filter(time == x)
    
    WT_tb <- WT_t %>% filter(dose == "Complex_2.5")
    
    WT_tb2 <- WT_t %>% filter(dose == y)
    
    res <- wilcox.test(WT_tb$value, WT_tb2$value, paired = FALSE)
    print("STARTING TEST FOR >>>>")
    print(x)
    print(y)
    print(res)
    
  }
  
  
}


###################### CLMM ############
# 
library(ordinal)

rad51 <- within(rad51, {   
  valueranked <- NA # need to initialize variable
  valueranked[value < 0.002] <- "1"
  valueranked[value >= 0.002 & value < 0.004] <- "2"
  valueranked[value >= 0.004 & value < 0.006] <- "3"
  valueranked[value >= 0.006 & value < 0.008] <- "4"
  valueranked[value >= 0.008 & value < 0.010] <- "5"
  valueranked[value >= 0.010 & value < 0.012] <- "6"
  valueranked[value >= 0.012 & value < 0.014] <- "7"
  valueranked[value >= 0.014 & value < 0.016] <- "8"
  valueranked[value >= 0.016 & value < 0.018] <- "9"
  valueranked[value >= 0.018 & value < 0.020] <- "10"
  valueranked[value >= 0.020 & value < 0.022] <- "11"
  valueranked[value >= 0.022] <- "12"
} )

View(rad51)
rad51$valueranked <-as.factor(rad51$valueranked)
mod = clmm(valueranked~dose*time+(1|id), data=rad51, Hess=T, nAGQ=17)

library(car)

library(RVAideMemoire)

Anova.clmm(mod,
           type = "II")

# PAIRED MEDIANS
install.packages("emmeans")
install.packages("multcomp")
library(emmeans)

marginal = emmeans(mod,
                   ~ dose + time)
pairs(marginal,
      adjust="tukey")


library(multcomp)

cld(marginal, Letters=letters)


################################

### Use Kruskal wallis
# Load necessary library
library(dplyr)

# Read the data from a CSV file
data <- read.delim("Half_lives_RM2ANOVA.tsv")

# Ensure the 'Half_life' column is a factor
data$Half_life <- as.factor(data$Half_life)

# Perform the Kruskal-Wallis test
kruskal_test <- kruskal.test(value ~ Half_life, data = data)

# Print the result of the Kruskal-Wallis test
print(kruskal_test)

# If significant, perform pairwise comparisons using Wilcoxon rank-sum tests with Holm correction
if (kruskal_test$p.value < 0.05) {
  pairwise_results <- pairwise.wilcox.test(data$value, data$Half_life, p.adjust.method = "holm")
  print(pairwise_results)
}


