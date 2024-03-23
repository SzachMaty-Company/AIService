data = []
with open("hh", "r") as f:
    for line in f:
        data.append(line.strip()[:4])

# for each country in the data, count how often it appears
country_counts = {}
for country in data:
    if country in country_counts:
        country_counts[country] += 1
    else:
        country_counts[country] = 1

# print stats ordered by values
sorted_items = sorted(country_counts.items(), key=lambda x: x[0])
print([int(x[1]) for x in sorted_items if int(x[0]) > 1998])
exit()
for key, value in sorted_items:
    if key >1998:
        print(key)
exit()
country_counts.pop("International")

# for each country print line with (lat, lng, count)
print([[*countries[country], count] for country, count in country_counts.items()])

23.52, 23.68, 20.65, 20.8, 19.62, 18.41, 18.09, 18.61, 19.5, 19.59, 19.9, 20.27, 20.37, 21.67, 25.06, 24.55, 24.37, 24.39, 24.33, 25.66, 26.2, 29.01, 19.65, 21.37, 23.68, 23.69, 20.04, 19.74, 19.98, 19.85, 20, 20.45, 21.35, 21.17, 21.03, 21.06, 21.08, 22.64, 23.1, 22.81, 22.73, 22.92, 22.83, 24.47
