library(tidyverse)
library(knitr)
library(ggplot2)
library(jsonlite)
library(gt)
library(gtsummary)
library(scales)
library(patchwork)
library(ggrepel)
library(tidygraph)
library(ggraph)
library(igraph)
library(dplyr)

#setwd('/work/UCDW')
setwd("/home/kgk/repos/bp_UCDW")

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

graph <- tbl_graph(edges = keyws_net_graph_df, directed = TRUE)

ggraph(graph, layout = 'fr') +
  geom_edge_link(aes(color = type), 
      arrow = arrow(length = unit(4, 'mm')), 
      end_cap = circle(3, 'mm'), 
      width = 1.2) +
  geom_node_point(size = 5, color = "black") +
  geom_node_text(aes(label = name), vjust = -1) +
  scale_edge_colour_brewer(type = "qual", palette = "Dark2") + 
  theme_light() +
  facet_wrap(~type)
  #labs(edge_color = "Relationship Type") +
  ggtitle("Custom-Coloured Network Visualization")


# Compute global layout
layout_all <- create_layout(graph, layout = 'fr')
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
      edges_long, to %in% nodes_d), 
    aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
    colour = "darkorange",
    arrow = arrow(angle = 20, length = unit(3, 'mm')), size = 0.5) +
  geom_point(data = 
    filter(nodes_df, name %in% nodes_d),
   aes(x = x, y = y), size = 4, color = "black") +
  geom_text(data = 
    filter(nodes_df, name %in% nodes_d), 
  aes(x = x, y = y, label = name), vjust = -1) +
  theme_void() #+
  #labs(title = "Faceted Network by Type", color = "Edge Type")

graph_vis_p <- ggplot() +
  geom_segment(
    data = filter(
      edges_long, to %in% nodes_p), 
    aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
    colour = "purple",
    arrow = arrow(angle = 20, length = unit(3, 'mm')), size = 0.5) +
  geom_point(data = 
    filter(nodes_df, name %in% nodes_p),
   aes(x = x, y = y), size = 4, color = "black") +
  geom_text(data = 
    filter(nodes_df, name %in% nodes_p), 
  aes(x = x, y = y, label = name), vjust = -1) +
  theme_void() #+
  #labs(title = "Faceted Network by Type", color = "Edge Type")

combined_plot = graph_vis_d + graph_vis_p
