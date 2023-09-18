def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def cf1(uid, sid, train_accepted):
    sxy, syx = 0, 0
        
    nei1 = train_accepted[train_accepted["source_id"] == sid]["user_id"].values
    nei2 = train_accepted[train_accepted["source_id"] == uid]["user_id"].values

    if len(nei1) > 0:
        for u in nei1:
            sxy += jaccard(train_accepted[train_accepted["source_id"] == u]["user_id"].values, nei2)
        sxy = sxy/len(nei1)

    if len(nei2) >0:
        for v in nei2:
            syx += jaccard(train_accepted[train_accepted["source_id"] == v]["user_id"].values, nei1)
        syx = syx/len(nei2)

    if sxy >0 and syx > 0:
        return 2/((1/sxy) +(1/syx))
    else:
        return 0

def cf2(uid, sid, train_accepted):
    sxy, syx = 0, 0
        
    nei1 = train_accepted[train_accepted["user_id"] == sid]["source_id"].values
    nei2 = train_accepted[train_accepted["user_id"] == uid]["source_id"].values

    if len(nei1) > 0:
        for u in nei1:
            sxy += jaccard(train_accepted[train_accepted["user_id"] == u]["source_id"].values, nei2)
        sxy = sxy/len(nei1)

    if len(nei2) >0:
        for v in nei2:
            syx += jaccard(train_accepted[train_accepted["user_id"] == v]["source_id"].values, nei1)
        syx = syx/len(nei2)

    if sxy >0 and syx > 0:
        return 2/((1/sxy) +(1/syx))
    else:
        return 0

def cf3(uid, sid, train_accepted):
    sxy, syx = 0, 0
        
    nei1 = train_accepted[train_accepted["user_id"] == sid]["source_id"].values
    nei2 = train_accepted[train_accepted["source_id"] == uid]["user_id"].values

    if len(nei1) > 0:
        for u in nei1:
            sxy += jaccard(train_accepted[train_accepted["source_id"] == u]["user_id"].values, nei2)
        sxy = sxy/len(nei1)

    if len(nei2) >0:
        for v in nei2:
            syx += jaccard(train_accepted[train_accepted["user_id"] == v]["source_id"].values, nei1)
        syx = syx/len(nei2)

    if sxy >0 and syx > 0:
        return 2/((1/sxy) +(1/syx))
    else:
        return 0

def cf4(uid, sid, train_accepted):
    sxy, syx = 0, 0
        
    nei1 = train_accepted[train_accepted["source_id"] == sid]["user_id"].values
    nei2 = train_accepted[train_accepted["user_id"] == uid]["source_id"].values

    if len(nei1) > 0:
        for u in nei1:
            sxy += jaccard(train_accepted[train_accepted["user_id"] == u]["source_id"].values, nei2)
        sxy = sxy/len(nei1)

    if len(nei2) >0:
        for v in nei2:
            syx += jaccard(train_accepted[train_accepted["source_id"] == v]["user_id"].values, nei1)
        syx = syx/len(nei2)

    if sxy >0 and syx > 0:
        return 2/((1/sxy) +(1/syx))
    else:
        return 0