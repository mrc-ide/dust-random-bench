#!/usr/bin/env Rscript

library(ggplot2)

dat <- read.csv("data/uniform.csv")

dir.create("figs", FALSE, TRUE)

## Setup cost shows linear increase for dust, nonlinear setup for curand
p_setup <- ggplot(dat, aes(x = n_threads, y = t_setup, group = engine)) +
  scale_x_continuous(trans = "log2") +
  scale_y_continuous(trans = "log10")+
  geom_point(aes(col = factor(engine))) +
  scale_colour_discrete(name = "Engine") +
  xlab("Number of threads") +
  ylab("Wall time (s)") +
  theme_bw()

ggsave("figs/setup.png", p_setup,
       width = 800, height = 500, units = "px", dpi = 100)

## Both generators show (unsurprisingly) generally linear growth in
## wall time for more samples, though the overhead for dust takes
## longer to pay off in small generators.
p_sample <-
  ggplot(dat, aes(x = n_draws, y = t_sample / n_draws, group = n_threads)) +
  scale_x_continuous(trans = "log2") +
  scale_y_continuous(trans = "log10")+
  geom_line(aes(col = factor(n_threads))) +
  facet_grid(~ engine) +
  scale_colour_discrete(name = "Threads") +
  xlab("Number of draws (per thread)") +
  ylab("Relative wall time (s)") +
  theme_bw()

ggsave("figs/sample.png", p_sample,
       width = 800, height = 500, units = "px", dpi = 100)

## Total hack here, someone good at ggplot will know how to do this in
## one shot:
dat_dust <- dat[dat$engine == "dust", ]
dat_curand <- dat[dat$engine == "curand", ]
dat_rel <- dat_dust
dat_rel$t_setup <- dat_dust$t_setup / dat_curand$t_setup
dat_rel$t_sample <- dat_dust$t_sample / dat_curand$t_sample
p_sample_rel <-
  ggplot(dat_rel, aes(x = n_draws, y = t_sample, group = n_threads)) +
  scale_x_continuous(trans = "log2") +
  scale_y_continuous(trans = "log10")+
  geom_line(aes(col = factor(n_threads))) +
  scale_colour_discrete(name = "Threads") +
  xlab("Number of draws (per thread)") +
  ylab("Relative wall time (dust / curand)") +
  theme_bw()

ggsave("figs/sample-rel.png", p_sample_rel,
       width = 800, height = 500, units = "px", dpi = 100)
