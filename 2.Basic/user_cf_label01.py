import sys
sys.path.append('../') # 到项目的根目录即可

from data_loader import dataloader
from data import filepaths as fp
from tqdm import tqdm #进度条的库
from utils import evaluate
import collections

#集合形式读取数据, 返回{uid1:{iid1,iid2,iid3}}
def getSet( triples ):
    # 用户喜欢的物品集 {uid:{iid1,..}}
    user_pos_items = collections.defaultdict(set)
    # 用户不喜欢的物品集 {uid:{iid1,..}}
    user_neg_items = collections.defaultdict(set)
    # 用户交互过的所有物品集（无论用户喜欢不喜欢 = 前两个的并集）
    user_all_items = collections.defaultdict(set)
    # 已物品为索引，喜欢物品的用户集 
    # {iid:{uid1,uid2,...}} 记录一个物品被哪些用户喜爱
    item_users = collections.defaultdict(set)
    
    for u, i, r in triples:
        user_all_items[u].add(i)
        if r == 1: # 用户喜欢
            user_pos_items[u].add(i)
            item_users[i].add(u)
        else:
            user_neg_items[u].add(i)
    return user_pos_items, item_users, user_neg_items, user_all_items

def knn4set(trainset, k, sim_method):
    '''
    ==TODO==
    :param trainset: 训练集合
    :param k: 近邻数量
    :param sim_method: 相似度方法
    :return: {样本1:[近邻1,近邻2，近邻3]}
    '''
    res = {}
    for e1 in tqdm(trainset):
        # 找到e1的k个邻居，这里的e1可以表示用户，也可以表示物品
        neighbors = []
        for e2 in trainset:
            if e1==e2 or len(trainset[e1]&trainset[e2])==0:
                continue
            sim = sim_method(trainset[e1],trainset[e2]) # 根据两者对应的集合衡量相似度
            neighbors.append((e2,sim))
        neighbors.sort(key=lambda x:x[1],reverse=True)
        res[e1] = [x[0] for x in neighbors[:k]] # 相似度最高的k个邻居
    return res

def get_recomedations_by_usrCF( user_sims, user_o_set ):
    '''
    ==TODO==
    :param user_sims: 用户的近邻集:{样本1:[近邻1,近邻2，近邻3]}
    :param user_o_set: 用户的原本喜欢的物品集合:{用户1:{物品1,物品2，物品3}} [但这里好像包括原先不喜欢的物品]
    :return: 每个用户的推荐列表{用户1:[物品1，物品2，物品3]}
    '''
    recomedations = collections.defaultdict(set)
    for user in user_sims:
        for sim_user in user_sims[user]:
            # 去除已经推荐过的电影做推荐
            recomedations[user] |= (user_o_set[sim_user]-user_o_set[user])
    return recomedations

def get_recomedations_by_itemCF(item_sims, user_o_set):
    '''
    ==TODO==
    :param item_sims: 物品的近邻集:{样本1:[近邻1,近邻2，近邻3]}
    :param user_o_set: 用户的原本喜欢的物品集合:{用户1:{物品1,物品2，物品3}}
    :return: 每个用户的推荐列表{用户1:[物品1，物品2，物品3]}
    '''
    recomedations = collections.defaultdict(set)
    for u in user_o_set:
        for item in user_o_set[u]:
            # 将自己喜欢物品的近邻物品与自己观看过的视频去重后推荐给自己
            if item in item_sims:
                recomedations[u] |= set( item_sims[item] ) - user_o_set[u]
    return recomedations

def trainUserCF( user_items_train, sim_method, user_all_items, k = 5 ):
    """
    ==TODO==
    """
    # 1. knn找到相似用户
    user_sims = knn4set( user_items_train, k, sim_method )
    # 2. 推荐物品
    recomedations = get_recomedations_by_usrCF( user_sims, user_all_items )
    return recomedations

def trainItemCF( item_users_train, sim_method, user_all_items, k = 5 ):
    """
    ==TODO==
    """
    # 1. knn找到相似物品
    item_sims = knn4set( item_users_train, k, sim_method )
    # 2. 根据用户喜欢的物品推荐相似物品
    recomedations = get_recomedations_by_itemCF( item_sims, user_all_items )
    return recomedations

def evaluation( test_set, user_neg_items, pred_set ):
    total_r = 0.0
    total_p = 0.0
    has_p_count = 0
    for uid in test_set:
        if len(test_set[uid]) > 0:
            # 把pred_set(用户喜欢的物品)做为正样本
            # 这里的test_set是正样本，user_neg_items是负样本
            p = evaluate.precision4Set( test_set[uid], user_neg_items[uid], pred_set[uid] )
            if p:
                total_p += p
                has_p_count += 1
            total_r += evaluate.recall4Set( test_set[uid], pred_set[uid] )
    print("Precision {:.4f} | Recall {:.4f}".format(total_p / has_p_count, total_r / len(test_set)))

if __name__ == '__main__':
    _, _, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1) # 87720  9746 (user_id,movie_id,0/1) 1 = like
    # user_pos_items, item_users, user_neg_items, user_all_items
    # user_pos_items + user_neg_items = user_all_items
    # 只用了正样本进行训练，这里的user_items_train和user_pos_items_test会有重合的user但是会多一些没有见过的物品
    user_items_train, item_users_train, _ ,user_all_items= getSet(train_set)
    user_pos_items_test, _, user_neg_items_test,_ = getSet(test_set)

    sim_method = lambda set1,set2:len(set1&set2)/((len(set1)*len(set2))**0.5) # cos4set
    
    print('user_CF')
    recomedations_by_userCF = trainUserCF( user_items_train, sim_method, user_all_items, k = 5 ) # user_items_train
    evaluation( user_pos_items_test, user_neg_items_test , recomedations_by_userCF )

    print('item_CF')
    recomedations_by_itemCF = trainItemCF( item_users_train, sim_method, user_all_items, k = 5 )
    evaluation( user_pos_items_test, user_neg_items_test , recomedations_by_itemCF )
