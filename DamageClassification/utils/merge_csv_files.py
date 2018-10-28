import csv


damage_path = '../data/damages.csv'
feature_path = '../data/features.csv'
data_merged_path = '../data/data_all.csv'

all_damages = dict()
all_features = dict()

with open(damage_path, 'r', newline='') as df:
    reader = csv.reader(df, delimiter=' ', quotechar='|')
    pre_visit = '0'
    damages = ''
    for row in reader:
        visit, damage = row[0].split(',')
        if visit == 'visit':
            continue

        if visit == pre_visit:
            if damages == '':
                damages = damage[-1]
            else:
                damages = damages + ', ' + damage[-1]
        else:
            all_damages[pre_visit] = damages
            pre_visit = visit
            damages = damage[-1]
    all_damages[visit] = damages

with open(feature_path, 'r', newline='') as ff:
        reader = csv.reader(ff, delimiter=' ', quotechar='|')
        pre_visit = '0'
        features = ''
        for row in reader:
            visit, feature = row[0].split(',')
            feature = feature.split('_')[-1]
            if visit == 'visit':
                continue

            if visit == pre_visit:
                if features == '':
                    features = feature
                else:
                    features = features + ', ' + feature
            else:
                all_features[pre_visit] = features
                pre_visit = visit
                features = feature
        all_features[visit] = features


with open(data_merged_path, 'w', newline='') as dmp:
    fieldnames = ['visit', 'damage', 'feature']
    writer = csv.DictWriter(dmp, fieldnames=fieldnames)
    writer.writeheader()

    for visit in all_damages:
        damage = all_damages[visit]
        feature = all_features[visit]

        writer.writerow({'visit': visit, 'damage': damage, 'feature': feature})

print('Successfully merged two files')

