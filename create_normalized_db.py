#!/usr/bin/env python

import sqlite3
import pdb
import json
import pickle as pkl
from utils.nlp import normalize 
from preprocess_data import get_db_pointer 
from utils.db_pointer import *
from tqdm import tqdm

DOMAINS = ['restaurant', 'hotel', 'attraction', 'train']
# 3 domains do not have a DB: 'taxi', 'hospital', 'police'
UNNORMALIZED_SLOTS = ['trainID', 'id', 'phone']
dbs = {}
dbs_conn = {}
for domain in DOMAINS:
    db = 'data2.0/db_normalized/{}-dbase.db'.format(domain)
    conn = sqlite3.connect(db)
    dbs[domain] = conn

for domain in DOMAINS: 
    query = 'select * from {}'.format(domain)
    cursor = dbs[domain].cursor().execute(query)
    entities = cursor.fetchall()
    print("Domain {} Num Entities {}".format(domain, len(entities)))
    keys = list(map(lambda x: x[0], cursor.description))
    print("Keys: {}".format(keys))
    if domain == 'train':
        id_index = keys.index('trainID')
    else:
        id_index = keys.index('id')
    all_ids = set([e[id_index] for e in entities])
    if len(all_ids) != len(entities):
        print("Domain {} entity IDs are not unique".format(domain))
    
    for entity in entities:
        update_query = 'UPDATE {} SET '.format(domain)
        values = ()
        for idx, v in enumerate(entity):
            if keys[idx] == 'id': 
                index = v
            elif keys[idx] == 'trainID':
                index = v
            if keys[idx] == 'arriveBy':
                arrive_by = v 
            if keys[idx] not in UNNORMALIZED_SLOTS:
                values = (*values, normalize(v))
                update_query += '{} = ? , '.format(keys[idx])
        values = (*values, index)
        update_query = update_query[:-2]
        if domain == 'train':
            update_query += 'WHERE trainID = ? AND arriveBy = ?'
            values = (*values, arrive_by)
        else:
            update_query += 'WHERE id = ?'
        dbs[domain].cursor().execute(update_query, values)
        dbs[domain].commit()

