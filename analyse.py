import os, csv, time, json, requests, logging
from datetime import datetime
from collections import defaultdict, deque
import pandas as pd
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("crawler.log", encoding="utf-8"), # Lưu log vào file
        logging.StreamHandler()                               # In log ra màn hình
    ]
)
logger = logging.getLogger(__name__)

# Targets sau k-core
FINAL_USERS_MIN  = 6_000
FINAL_INT_MIN    = 1_200_000

# Crawl buffer
CRAWL_USERS_TARGET = 35_000
CRAWL_INT_TARGET   = 3_500_000

# K-core params
MIN_INTERACTIONS = 5
MIN_USER_PER_ITEM = 5

# API page limits
MAX_STARRED_PAGES = 30
MAX_SEARCH_PAGES  = 10

# Dirs
OUT_DIR      = "output_github_v3"
RAW_DIR      = os.path.join(OUT_DIR, "raw")
LIGHTGCN_DIR = os.path.join(OUT_DIR, "lightgcn")
ULTRAGCN_DIR = os.path.join(OUT_DIR, "ultragcn")
IMREC_DIR    = os.path.join(OUT_DIR, "imrec")
PROFILES_DIR = os.path.join(OUT_DIR, "profiles")
CHECKPOINT   = os.path.join(OUT_DIR, "checkpoint.json")
RAW_CSV      = os.path.join(RAW_DIR, "interactions.csv")
RAW_REPO_CSV = os.path.join(RAW_DIR, "repos.csv")

for d in [OUT_DIR, RAW_DIR, LIGHTGCN_DIR, ULTRAGCN_DIR, IMREC_DIR, PROFILES_DIR]:
    os.makedirs(d, exist_ok=True)
def clean(interactions_df, repos_df, min_stars=10):
    # Chuẩn hóa case
    interactions_df = interactions_df.copy()
    repos_df = repos_df.copy()
    interactions_df['user'] = interactions_df['user'].str.lower().str.strip()
    interactions_df['repo'] = interactions_df['repo'].str.lower().str.strip()
    repos_df['repo'] = repos_df['repo'].str.lower().str.strip()

    # Lọc repo hợp lệ
    valid_repos = set(repos_df[
        (repos_df['language'] != 'N/A') &
        (repos_df['stars'] >= min_stars)
    ]['repo'])

    df = interactions_df[interactions_df['repo'].isin(valid_repos)].copy()

    # Loại self-star
    df = df[df.apply(lambda r: r['repo'].split('/')[0] != r['user'], axis=1)]

    # Loại duplicate, giữ timestamp sớm nhất
    df = df.sort_values('timestamp').drop_duplicates(
        subset=['user', 'repo'], keep='first'
    )

    return df
def build_interaction_dict(df):
    """Chuyển DataFrame → dict {user: {repo: timestamp}} để đưa vào kcore()"""
    data = defaultdict(dict)
    for _, row in df.iterrows():
        data[row['user']][row['repo']] = row['timestamp']
    return dict(data)

def load_raw(path):
    data = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            data[r["user"]][r["repo"]] = float(r["timestamp"])
    return data

def kcore(data):
    logger.info("--- PHASE 3: Đang lọc K-core ---")
    it = 0
    while True:
        prev_int = sum(len(v) for v in data.values())
        data = {u: i for u, i in data.items() if len(i) >= MIN_INTERACTIONS}
        item_count = defaultdict(int)
        for items in data.values():
            for i in items: item_count[i] += 1
        data = {u: {i: ts for i, ts in items.items() if item_count[i] >= MIN_USER_PER_ITEM} for u, items in data.items()}
        data = {u: i for u, i in data.items() if len(i) >= MIN_INTERACTIONS}
        curr_int = sum(len(v) for v in data.values())
        it += 1
        logger.info(f"Lần {it}: Còn {len(data):,} user | {curr_int:,} tương tác")
        if curr_int == prev_int: break
    return data

def remap(data):
    user2id = {u: i for i, u in enumerate(sorted(data.keys()))}
    items   = sorted(set(i for d in data.values() for i in d))
    item2id = {i: idx for idx, i in enumerate(items)}
    remapped = {}
    for u, its in data.items():
        sorted_items = sorted(its.items(), key=lambda x: x[1])
        remapped[user2id[u]] = [(item2id[i], ts) for i, ts in sorted_items]
    return user2id, item2id, remapped

def split(remapped):
    train, test = {}, {}
    for u, items in remapped.items():
        if len(items) <= 1:
            train[u] = [x[0] for x in items]; test[u] = []
        else:
            train[u] = [x[0] for x in items[:-1]]; test[u] = [items[-1][0]]
    return train, test

def export_all(train, test, user2id, item2id, remapped, repo_csv):
    logger.info("--- PHASE 4: Xuất file mô hình ---")
    for d_path in [LIGHTGCN_DIR, ULTRAGCN_DIR]:
        with open(os.path.join(d_path, "train.txt"), "w") as f:
            for u, items in train.items(): f.write(f"{u} {' '.join(map(str, items))}\n")
        with open(os.path.join(d_path, "test.txt"), "w") as f:
            for u, items in test.items(): f.write(f"{u} {' '.join(map(str, items))}\n")

    with open(os.path.join(IMREC_DIR, "github.inter"), "w") as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for u, items in remapped.items():
            for i, ts in items: f.write(f"{u}\t{i}\t{ts:.1f}\n")
    logger.info(f"Đã xuất xong tại thư mục: {OUT_DIR}")
def print_stats(user2id, item2id, remapped, train, test):
    n_u  = len(user2id)
    n_i  = len(item2id)
    n_t  = sum(len(v) for v in remapped.values())
    n_tr = sum(len(v) for v in train.values())
    n_te = sum(len(v) for v in test.values())
    density = n_t / (n_u * n_i) if n_u * n_i > 0 else 0

    logger.info("=" * 55)
    logger.info(f"  #Users        : {n_u:>10,}  ({'OK' if n_u >= FINAL_USERS_MIN else 'FAIL -- need more'})")
    logger.info(f"  #Items        : {n_i:>10,}")
    logger.info(f"  #Interactions : {n_t:>10,}  ({'OK' if n_t >= FINAL_INT_MIN else 'FAIL -- need more'})")
    logger.info(f"  #Train        : {n_tr:>10,}")
    logger.info(f"  #Test         : {n_te:>10,}")
    logger.info(f"  Density       : {density:>10.5f}")
    logger.info(f"  Avg int/user  : {n_t/n_u:>10.1f}")
    logger.info("=" * 55)
    
if __name__ == "__main__":
  
    raw_interaction = os.path.join("data", "interactions.csv")
    raw_repo = os.path.join("data", "repos.csv")
    if not os.path.exists(raw_interaction) or not os.path.exists(raw_repo):
        logger.error("Không tìm thấy file CSV! Hãy kiểm tra lại đường dẫn.")

    interactions_df = pd.read_csv(raw_interaction)
    repos_df = pd.read_csv(raw_repo)
    repos_df = repos_df.drop(columns=['description'])

    df_clean   = clean(interactions_df, repos_df, min_stars=10)
    data  = build_interaction_dict(df_clean)
    
    data = kcore(data )
    user2id, item2id, remapped = remap(data)
    train, test = split(remapped)

    export_all(train, test, user2id, item2id, remapped, raw_repo)
    print_stats(user2id, item2id, remapped, train, test)

    logger.info("hoàn thành")