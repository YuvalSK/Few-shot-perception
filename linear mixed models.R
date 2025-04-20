library(readr)
library(dplyr)
library(tidyr)
library(lme4)
library(splines)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(ggpubr)
library(sjPlot)
library(effects)

setwd("C://Users//User//Desktop//OneDrive - Bar Ilan University//Academic//PhD_Thesis//3 - individual differences//One shot//Data//Experiment 9 - preregistered")

df_estimates <- read_csv("merged - filtered - long.csv")

df_chng <- read_csv("merged - filtered.csv") %>%
  select(
    "indv", "STP", "prev", "Condition", "participant #"
  ) %>%
  filter(!Condition %in% c(6,7,8),
         STP != 'Internal')


df_chng$`participant` <- as.factor(df_chng$`participant #`)
df_chng$STP <- as.factor(df_chng$STP)
df_chng$prev <- as.factor(df_chng$prev)
df_chng$Condition <- as.factor(df_chng$Condition)

summary(df_chng)


mdl_1 <- lm(indv ~ prev,
              data = df_chng)
summary(mdl_1)
em <- emmeans(mdl_1, ~ prev)
contrast(em, "pairwise", adjust = "Bonferroni")

plot_model(mdl_1, type = "eff", terms = "prev")


mdl_2 <- lm(indv ~ prev*STP,
            data = df_chng)
summary(mdl_2)
anova(mdl_2)
em <- emmeans(mdl_2, ~ STP)
em
contrast(em, "pairwise", adjust = "Bonferroni")

#plot_model(mdl_2, type = "eff", terms = "STP")


mdl_4 <- lm(indv ~ prev + STP,
            data = df_chng)
summary(mdl_4)
anova(mdl_4)
em <- emmeans(mdl_4, ~ STP)
em
contrast(em, "pairwise", adjust = "Bonferroni")

#plot_model(mdl_4, type = "eff", terms = "STP")



mdl_3 <- lmer(indv ~ STP + (1|Condition),
              data = df_chng)
summary(mdl_3)
anova(mdl_3)
em <- emmeans(mdl_3, ~ STP)
em
contrast(em, "pairwise", adjust = "Bonferroni")
plot_model(mdl_3, type = "eff", terms = "STP")

fix_only_mdl <- lmer(indv ~ STP + (1|Condition),
                   data = subset(df_chng, df_chng$prev == "fixed"))
summary(fix_only_mdl)
anova(fix_only_mdl)



down_only_mdl <- lm(indv ~ STP,
                   data = subset(df_chng, df_chng$prev == "down"))
summary(down_only_mdl)
anova(down_only_mdl)
#plot_model(down_only_mdl, type = "eff", terms = "STP")
em <- emmeans(down_only_mdl, ~ STP)
em
contrast(em, "pairwise", adjust = "Bonferroni")
plot_model(down_only_mdl, type = "eff", terms = "STP")

up_only_mdl <- lm(indv ~ STP,
                      data = subset(df_chng, df_chng$prev == "up"))
summary(up_only_mdl)
anova(up_only_mdl)
plot_model(up_only_mdl, type = "eff", terms = "STP")
em <- emmeans(up_only_mdl, ~ STP)
em
contrast(em, "pairwise", adjust = "Bonferroni")


mdl_5 <- lm(indv ~ prev*STP,
            data = df_chng)
summary(mdl_5)

plot(mdl_5)


