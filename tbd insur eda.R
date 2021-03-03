library(data.table)
library(ggplot2)
library(DT)
dt = fread('data/travel_insurance.csv')
dt[, claim_bin := ifelse(claim == 'Yes', 1,0)]
# look at ration of claims
dt[, table(claim_bin)/sum(table(claim_bin))*100]

# Only 1.46% of polcies are claims. 

#Claims by region
claims_dest = dt[,.(
  num_claims = sum(claim_bin),
  claims_per_policy = sum(claim_bin) / .N 
  ), by = destination][num_claims > 0 ]
ggplot(claims_dest, aes(x=destination, y=num_claims, fill=destination)) +
  geom_bar(stat="identity")+theme_minimal()+theme(legend.position = "none")

#Singapore has a lot of claims!!
datatable(claims_dest)
datatable(claims_dest %>% setorder(-num_claims))

# Claims by Gender
ggplot(dt[,.(num_claims = sum(claim_bin)), by = gender], aes(x=gender, y=num_claims, fill=gender)) +
  geom_bar(stat="identity")+theme_minimal()+theme(legend.position = "none")

unique(dt$gender)

ggplot(claims_dest, aes(x="", y=num_claims, fill=destination)) +
  geom_bar(stat="identity", width=1) +
  coord_polar("y", start=0)+theme(legend.position = "none")

dt[duration == 0, duration := mean(duration)]

