def merge_add(old, new):
    for k in new.keys():
        if k in old: old[k] = [vo + vn for vo, vn in zip(old[k], new[k])]
        else: old[k] = new[k]
    return old


def merge_average(res, num_batch):
    for k in res.keys():
        res[k] = [v / num_batch for v in res[k]]
    return res