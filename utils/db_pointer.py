import sqlite3

import numpy as np
import pdb 
from utils.nlp import normalize


# loading databases
domains = ['restaurant', 'hotel', 'attraction', 'train']
# 3 domains do not have DB: 'taxi', 'hospital', 'police']

dbs = {}
for domain in domains:
    db = 'data2.0/db/{}-dbase.db'.format(domain)
    print("Connect to DB {}".format(db))
    conn = sqlite3.connect(db)
    c = conn.cursor()
    dbs[domain] = c

normalized_dbs = {}
for domain in domains:
    db = 'data2.0/db_normalized/{}-dbase.db'.format(domain)
    print("Connect to DB {}".format(db))
    conn = sqlite3.connect(db)
    c = conn.cursor()
    normalized_dbs[domain] = c
   

def get_db_columns(normalized):
  if normalized:
    database = normalized_dbs
  else:
    database = dbs 
  out = {}
  for domain, db in database.items():
      query = 'select * from {}'.format(domain)
      cursor = database[domain].execute(query)
      keys = list(map(lambda x: x[0], cursor.description))
      out[domain] = keys
  return out 

def get_all_entities(normalized):
  if normalized:
      database = normalized_dbs
  else:
      database = dbs 
  out = {}
  for domain, db in database.items():
      query = 'select * from {}'.format(domain)
      cursor = database[domain].execute(query)
      out[domain] = cursor.fetchall()
  return out 

normalized_dbs_columns = get_db_columns(True)
dbs_columns = get_db_columns(False)
assert normalized_dbs_columns == dbs_columns


def one_hot_vector(num, domain, vector):
    """Return number of available entities for particular domain."""
    number_of_options = 6
    if domain != 'train':
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0,0])
        elif num == 1:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    else:
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num > 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

    return vector

def query_result(domain, metadata, normalized):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    sql_query = "select * from {}".format(domain)
    flag = True
    if domain in metadata:
        if 'semi' in metadata[domain]:
            state = metadata[domain]['semi']
        else:
            state = metadata[domain]
    else:
        state = {} 
    for key, val in state.items():
        if 'fail_' in key: continue 
        if normalized:
            if key not in normalized_dbs_columns[domain]: pdb.set_trace()
        else:
            if key not in dbs_columns[domain]: pdb.set_trace()
        if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
            continue 
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"
    if normalized:
        entities = normalized_dbs[domain].execute(sql_query).fetchall()
    else:
        entities = dbs[domain].execute(sql_query).fetchall()
    return entities, len(entities)


def queryResultVenues(domain, turn, real_belief=False):
    # query the db
    sql_query = "select * from {}".format(domain)

    if real_belief == True:
        items = turn.items()
    elif real_belief=='tracking':
        for slot in turn[domain]:
            key = slot[0].split("-")[1]
            val = slot[0].split("-")[2]
            if key == "price range":
                key = "pricerange"
            elif key == "leave at":
                key = "leaveAt"
            elif key == "arrive by":
                key = "arriveBy"
            if val == "do n't care":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

            try:  # "select * from attraction  where name = 'queens college'"
                return dbs[domain].execute(sql_query).fetchall()
            except:
                return []  # TODO test it
        pass
    else:
        items = turn['metadata'][domain]['semi'].items()

    flag = True
    for key, val in items:
        if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " +key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    try:  # "select * from attraction  where name = 'queens college'"
        return dbs[domain].execute(sql_query).fetchall()
    except:
        return []  # TODO test it
