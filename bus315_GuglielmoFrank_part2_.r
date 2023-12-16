#### Association Rule Mining on the Mushroom Dataset ####

# Step 1: Install and Load R Packages
# Install arules and arulesViz if you haven't already
install.packages("arules")
install.packages("arulesViz")
# Load the packages
library(arules)
library(arulesViz)

# Step 2: Load the Mushroom Dataset
data(package = "arules")
data("Mushroom")

labels <- itemLabels(Mushroom)
print(labels)

help("Mushroom") #learn more about the data in Help

# Step 3: Investigate and Review Your Data
class(Mushroom)
arules::summary(Mushroom)
# use inspect() to review records (transactions - itemsets and their items)
inspect(Mushroom) #too many to show
inspect(head(Mushroom, 3)) #the first 3 entries
inspect(tail(Mushroom, 3)) #the last 3 entries

LIST(head(Mushroom)) #list view of the first 6 entries with items numbered
LIST(tail(Mushroom)) #list view of the last 6 entries with items numbered

frequent.items <- eclat(Mushroom) #default support is 10%-20% # nolint

frequent.items <- eclat(Mushroom, parameter = list(support = .25)) #min support # nolint
inspect(head(frequent.items), 3) #use inspect() to retrieve the frequent itemsets # nolint
frequent.items.count <- sort(frequent.items, by = "count") # nolint
inspect(head(frequent.items.count, 3))
inspect(tail(frequent.items.count, 3)) #the 3 items with the hishest support
#the following sorts in ascending order with decreasing=FALSE.
frequent.items.support <- sort(frequent.items, by = "support", decreasing = FALSE) # nolint
inspect(head(frequent.items.support, 3)) #the 3 items with the smallest support
inspect(tail(frequent.items.support, 3)) #the 3 items with the highest support

# visualize the most frequent items by using itemfrequencyPlot().
itemFrequencyPlot(Mushroom) #too many - filter the results
itemFrequencyPlot(Mushroom, topN = 10) #top 10 most frequent items
itemFrequencyPlot(Mushroom, topN = 10, type = "absolute") #by adding type="absolute" will # nolint
#see the absolute frequency values of the most frequent items in the plot.

# Step 4: Generate Association Rules
help("apriori") #read more about the apriori algorithm in Help.

rules <- apriori(Mushroom, parameter = list(support = .5, confidence = .5))
rules #0 rules are generated. We should increase the min support to identify more rules # nolint
rules <- apriori(Mushroom, parameter = list(support = .4, confidence = .8))
rules #15 rules are generates at 1% support and 50% confidence
#let's looks at the 15 rules
inspect(rules)

rules <- apriori(Mushroom,
                 parameter = list(support = .5,
                                  confidence = .7,
                                  minlen = 5,
                                  maxlen = 8))


rules.subset.support <- subset(rules, support<.75) #a subset of rules with more than a 1 lift # nolint

#visualize the rules using the graph technique
plot(rules) #scatter plot
plot(rules, method = "graph") #graph plot

rules.poison.rhs <- apriori(Mushroom, parameter = list(support=.4, confidence=.5),minlen=4,maxlen=6, # nolint
                      appearance = list(default = "lhs", rhs="Class=poisonous")) # nolint

rules.edible.rhs <- apriori(Mushroom, parameter = list(support=.4, confidence=.5), minlen=4,maxlen=6, # nolint
                      appearance = list(default = "lhs", rhs="Class=edible")) # nolint

#Sort by lift
rules.poison.rhs <- sort(rules.poison.rhs, by="lift") # nolint
rules.edible.rhs <- sort(rules.edible.rhs, by="lift") # nolint

inspect(head(rules.poison.rhs, 10))
inspect(head(rules.edible.rhs, 3))
inspect(tail(rules.poison.rhs, 3))
inspect(tail(rules.edible.rhs, 3))


plot(rules.poison.rhs) #scatter plot
plot(rules.poison.rhs, method = "graph") #graph plot

#### CLUSTER ANALYSIS ####

#### install and load packages ####
#skip installation if already installed
install.packages("cluster")
install.packages("clustMixType") #to cluster mixed data sets
install.packages("factoextra")
install.packages("dbscan")
install.packages("pastecs")

#load the installed packages
library(clustMixType)
library(factoextra)
library(cluster)
library(dbscan)
library(pastecs)

#### load data ####
help("votes.repub") #learn about the data
data("votes.repub") #load the data into R

#### learn about data ####
View(votes.repub)
summary(votes.repub) #attributes min mix mean median
str(votes.repub)
pastecs::stat.desc(votes.repub) #descriptive stats
class(votes.repub) #it is a data frame

head(votes.repub)
tail(votes.repub)

#### pre-processing ####
sum(is.na(votes.repub)) #there are 217 missing values
votes.repub = na.omit(votes.repub) #remove 217 data objects with missing values # nolint

votes.dist = get_dist(votes.repub, method = 'pearson') # nolint

fviz_dist(votes.dist) #visualize distance matrix

#what happens to total WSS when k in kmeans varies?
fviz_nbclust(votes.repub, kmeans, method = 'wss') #at the elbow, k values of 4, 5, and 6 could be optimal candidates # nolint


#k=4 # nolint
set.seed(123)
k4 <- kmeans(votes.repub, centers = 4, nstart = 25)
k4 #kmeans clusters information
k4$cluster
sort(k4$cluster) #sort k4$cluster to see the cluster membership orders by cluster number # nolint
k4$betweenss
k4$withinss

#k=5
set.seed(123)
k5 = kmeans(votes.repub, centers = 5, nstart = 25)
k5
sort(k5$cluster)
k5$betweenss
k5$withinss 


#k=6
set.seed(123)
k6 = kmeans(votes.repub, centers = 6, nstart = 25)
k6
sort(k6$cluster)
k6$betweenss 
k6$withinss

#k=7
set.seed(123)
k7 = kmeans(votes.repub, centers = 7, nstart = 25)
k7
sort(k7$cluster)
k7$betweenss
k7$withinss



#the smallest withinSS is preferred which means data object more similar to each other within any given cluster
#and the cluster is more dense and coherent

k4$withinss
k5$withinss
k6$withinss
k7$withinss

#you can see that as we increase k in kmeans, the withinSS for some clusters become zero
#which means only one data object in those clusters. This is not insightful if we only
#have one data object in a cluster. kmeans is not a good clustering choice for this data.

#you can look into each cluster by filtering your data with the cluster number
#for example, I want to know which objects are in cluster 1 in k4
votes.repub[k4$cluster==1,]

votes.repub[k4$cluster==2,]

#visualize kmeans clusters
fviz_cluster(k4, data = votes.repub) 
fviz_cluster(k5, data = votes.repub)




#single
hc.single = agnes(votes.repub, method = 'single')
hc.single$ac #0.507

hc.complete = agnes(votes.repub, method = 'complete')
hc.complete$ac #0.7972274

hc.average = agnes(votes.repub, method = 'average')
hc.average$ac #0.67

plot(hc.complete)

