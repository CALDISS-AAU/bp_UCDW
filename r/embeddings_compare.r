library(tidyverse)
library(knitr)
library(ggplot2)
library(jsonlite)
#library(gt)
#library(gtsummary)
library(scales)
library(patchwork)
library(ggrepel)
library(tidygraph)
library(ggraph)
library(igraph)
library(dplyr)
library(ggtext)
library(autograph)

setwd('/work/UCDW')
#setwd("/home/kgk/repos/bp_UCDW")

## plot data
keyws_proj <- read_csv('output/embeddings/keyword_projections.csv')
keyws_net <- read_csv('output/embeddings/plotting_data_keywords_compare_gender_network.csv')

## theme
theme_use <- theme_minimal(base_family = "serif",
                           base_size = 15)


## PROJECTIONS
# normalize x and y
keyws_proj <- keyws_proj |> 
  group_by(model) |> 
  mutate(
    min_model_x = min(x),
    max_model_x = max(x),
    min_model_y = min(y),
    max_model_y = max(y)) |> 
  ungroup() |> 
  mutate(
    x_norm = (x - min_model_x) / (max_model_x - min_model_x),
    y_norm = (y - min_model_y) / (max_model_y - min_model_y))


keyws_proj |> 
  mutate(type = case_match(
    model, 
    "model_a" ~ "Al materiale",
    "model_f" ~ "Piger",
    "model_m" ~ "Drenge"
  )) |> 
  filter(model != "model_a") |> 
  ggplot(aes(x = x_norm, y = y_norm, label = keyword, colour = type)) + 
  geom_point(size = 0.5) + 
  geom_text(
    nudge_y = 0.03
  ) + 
  scale_colour_brewer(type = "qual", palette = "Dark2") + 
  labs(colour = "") + 
  facet_wrap(~type) + 
  theme_use

## NETWORK
keyws_net_graph_df <- keyws_net |> 
  filter(!is_keyword) |> 
  rename(
    from = keyword,
    to = word
  ) |> 
  mutate(type = case_match(
    model, 
    "model_f" ~ "Piger",
    "model_m" ~ "Drenge"
  )) |> 
  select(from, to, type)

# words to exclude in labels
labels_exclude <- c(
  # almindelige funktions-/fyldord
  "bare", "helt", "dertil", "efterhånden", "endelig", "muligt",
  "modsat", "korrekt",
  
  # tal / ord for tal
  "12.", "9.", "7.", "syvende", "halvanden",
  
  # forkortelser / symbol-lignende
  "f.x",
  
  # meget generelle, lavt informationsindhold
  "nylig", "april", "imorges", "folk"
)

# keywords used for finding neighbours
keyws_main <- unique(keyws_net$keyword)

# create graph
graph <- tbl_graph(edges = keyws_net_graph_df, directed = TRUE)

# Compute global layout
layout_all <- create_layout(graph, layout = 'stress')
nodes_df <- layout_all %>% as_tibble() %>% select(name, x, y)

# filter edges
nodes_d <- keyws_net_graph_df |> 
  filter(type == "Drenge") |> 
  select(to, from) |> 
  pivot_longer(everything()) |> 
  pull(value)

nodes_p <- keyws_net_graph_df |> 
  filter(type == "Piger") |> 
  select(to, from) |> 
  pivot_longer(everything()) |> 
  pull(value)

# Join to get edge coordinates
edges_long <- keyws_net_graph_df %>%
  left_join(nodes_df, by = c("from" = "name")) %>%
  rename(x_from = x, y_from = y) %>%
  left_join(nodes_df, by = c("to" = "name")) %>%
  rename(x_to = x, y_to = y)

# Plots
graph_vis_d <- ggplot() +
  geom_segment(
    data = filter(
      edges_long, to %in% nodes_d, !to %in% labels_exclude), 
    aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
    colour = "darkorange",
    arrow = arrow(angle = 20, length = unit(3, 'mm')), size = 0.5) +
  geom_point(data = 
    filter(nodes_df, name %in% nodes_d, !name %in% labels_exclude),
   aes(x = x, y = y), size = 2, color = "black") +
  geom_text_repel(data = 
    filter(nodes_df, name %in% nodes_d, !name %in% labels_exclude, !name %in% keyws_main), 
  aes(x = x, y = y, label = name), max.overlaps = 10) +
  geom_label_repel(data = 
    filter(nodes_df, name %in% keyws_main), 
  aes(x = x, y = y, label = name), max.overlaps = 10) +
  theme_void() +
  labs(title = "Drenge") + 
  theme(
    plot.title = element_textbox(
      fill = "darkgrey", 
      colour = "white", 
      box.colour = "black",
      hjust = 0.5,
      face = "bold",
      size = 14,
      linetype = "solid",
      padding = unit(c(0.02,0.15,0.02,0.15), "npc")
  ))
  

graph_vis_p <- ggplot() +
  geom_segment(
    data = filter(
      edges_long, to %in% nodes_p, !to %in% labels_exclude), 
    aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
    colour = "purple",
    arrow = arrow(angle = 20, length = unit(3, 'mm')), size = 0.5) +
  geom_point(
    data = filter(nodes_df, name %in% nodes_p, !name %in% labels_exclude),
   aes(x = x, y = y), size = 2, color = "black") +
  geom_text_repel(data = 
    filter(nodes_df, name %in% nodes_p, !name %in% labels_exclude, !name %in% keyws_main), 
  aes(x = x, y = y, label = name), max.overlaps = 10) +
  geom_label_repel(data = 
    filter(nodes_df, name %in% keyws_main), 
  aes(x = x, y = y, label = name), max.overlaps = 10) +
  theme_void() +
  labs(title = "Piger") + 
  theme(
    plot.title = element_textbox(
      fill = "darkgrey", 
      colour = "white", 
      box.colour = "black",
      hjust = 0.5,
      face = "bold",
      size = 14,
      linetype = "solid",
      padding = unit(c(0.02,0.15,0.02,0.15), "npc")
  ))

combined_plot <- graph_vis_d + graph_vis_p

ggsave(
  file.path('output', 'plots', 'embeddings_word-relations_compare.png'),
  plot = combined_plot,
  width = 8,
  height = 5,
  unit = "in",
  device = "png",
  dpi = 300,
  scale = 1.5
)
