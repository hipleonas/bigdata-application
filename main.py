import os, csv, time, json, requests, logging
from datetime import datetime
from collections import defaultdict, deque
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
# ─────────────────────────────────────────────
# CẤU HÌNH LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("crawler.log", encoding="utf-8"), # Lưu log vào file
        logging.StreamHandler()                               # In log ra màn hình
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG NÂNG CẤP (Mục tiêu 1.2M+ Interactions)
# ─────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("PAS_MAIN")
BASE_URL = "https://api.github.com"

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

# Interaction type weights for scoring (used in export/model)
INTERACTION_WEIGHTS = {
    "star":    1.0,
    "fork":    1.5,   # stronger signal than star
    "watch":   0.8,
    "follow":  0.5,   # user→user signal (mapped to user as "item")
}
# Toggle which signals to collect
COLLECT_STARS   = True
COLLECT_FORKS   = True
COLLECT_WATCHES = True
COLLECT_FOLLOWS = False  # user-follow graph is separate; enable if needed

for d in [OUT_DIR, RAW_DIR, LIGHTGCN_DIR, ULTRAGCN_DIR, IMREC_DIR, PROFILES_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# SMART QUERY GENERATOR
# ─────────────────────────────────────────────
def generate_smart_queries():
    langs = ["python", "javascript", "java", "go", "rust", "typescript", "cpp", "php", "ruby", "swift"]
    ranges = [
        (50, 65), (66, 80), (81, 95), (96, 110), (111, 130),
        (131, 155), (156, 185), (186, 220), (221, 260), (261, 310),
        (311, 400), (401, 600), (601, 1000), (1001, 5000)
    ]
    queries = []
    for lang in langs:
        for low, high in ranges:
            queries.append(f"followers:{low}..{high} language:{lang}")
    return queries

# ─────────────────────────────────────────────
# REQUEST HELPERS
# ─────────────────────────────────────────────
def _handle_rate_limit(r, is_search=False):
    remaining = int(r.headers.get("X-RateLimit-Remaining", 999))
    reset_at  = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))

    if r.status_code in [403, 429]:
        wait = max(reset_at - int(time.time()), 30) + 5
        logger.warning(f"Rate limit! Ngủ {wait}s...")
        time.sleep(wait)
        return False

    if r.status_code == 200:
        if is_search:
            time.sleep(2.2)
            if remaining < 5:
                wait = max(reset_at - int(time.time()), 10) + 3
                time.sleep(wait)
        else:
            if remaining < 100:
                wait = max(reset_at - int(time.time()), 10) + 2
                time.sleep(wait)
    return True

def _get_rest(url, params=None, star_header=False):
    accept = "application/vnd.github.v3.star+json" if star_header else "application/vnd.github+json"
    for attempt in range(5):
        try:
            r = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": accept}, params=params, timeout=15)
            if not _handle_rate_limit(r, is_search=False): continue
            return r.json() if r.status_code == 200 else []
        except Exception as e:
            time.sleep(5)
    return []

def _get_search(url, params=None):
    for attempt in range(5):
        try:
            r = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}, params=params, timeout=15)
            if not _handle_rate_limit(r, is_search=True): continue
            return r.json() if r.status_code == 200 else {}
        except Exception as e:
            time.sleep(5)
    return {}

# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
def save_cp(data: dict):
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f)

def load_cp() -> dict:
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}

# ─────────────────────────────────────────────
# PHASE 1: Collect users (Smart Strategy)
# ─────────────────────────────────────────────
def collect_users(cp: dict):
    if cp.get("phase") in ["crawl", "process", "done"]:
        return cp["users"]

    users = list(cp.get("users", []))
    done_queries = set(cp.get("done_queries", []))
    visited = set(users)
    SEARCH_QUERIES = generate_smart_queries()

    logger.info(f"--- PHASE 1: Tìm {CRAWL_USERS_TARGET} User chất lượng ---")
    for query in SEARCH_QUERIES:
        if query in done_queries or len(users) >= CRAWL_USERS_TARGET: continue

        for page in range(1, MAX_SEARCH_PAGES + 1):
            if len(users) >= CRAWL_USERS_TARGET: break
            data = _get_search(f"{BASE_URL}/search/users", {"q": query, "per_page": 100, "page": page, "sort": "repositories"})
            items = data.get("items", []) if isinstance(data, dict) else []
            if not items: break

            for u in items:
                login = u.get("login")
                if login and login not in visited:
                    visited.add(login); users.append(login)

            logger.info(f"Query: {query[:35]}... | Đã lấy: {len(users):,}")
            if len(items) < 100: break

        done_queries.add(query)
        save_cp({"phase": "collect", "users": users, "done_queries": list(done_queries)})

    logger.info(f"DONE Phase 1: {len(users):,} users.")
    save_cp({"phase": "crawl", "users": users, "crawl_idx": 0, "total_interactions": 0})
    return users

# ─────────────────────────────────────────────
# PHASE 2: Crawl multi-signal interactions
# ─────────────────────────────────────────────
def _crawl_starred(u, seen_repos, w_repo):
    """Returns list of (user, repo, timestamp, 'star')"""
    results = []
    for page in range(1, MAX_STARRED_PAGES + 1):
        data = _get_rest(f"{BASE_URL}/users/{u}/starred", {"per_page": 100, "page": page}, star_header=True)
        if not data or not isinstance(data, list): break
        for item in data:
            repo = item["repo"]
            name = repo.get("full_name")
            if not name: continue
            ts = int(datetime.strptime(item["starred_at"], "%Y-%m-%dT%H:%M:%SZ").timestamp())
            results.append((u, name, ts, "star"))
            if name not in seen_repos:
                seen_repos.add(name)
                w_repo.writerow([name, repo.get("language") or "N/A", repo.get("stargazers_count", 0), ""])
        if len(data) < 100: break
    return results

def _crawl_forks(u, seen_repos, w_repo):
    """Returns list of (user, repo, timestamp, 'fork') from user's forked repos."""
    results = []
    for page in range(1, MAX_STARRED_PAGES + 1):
        data = _get_rest(f"{BASE_URL}/users/{u}/repos", {"per_page": 100, "page": page, "type": "forks"})
        if not data or not isinstance(data, list): break
        for repo in data:
            if not repo.get("fork"): continue
            parent = repo.get("parent") or {}
            name = parent.get("full_name") or repo.get("full_name")
            if not name: continue
            ts_str = repo.get("created_at", "")
            try:
                ts = int(datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").timestamp())
            except Exception:
                ts = int(time.time())
            results.append((u, name, ts, "fork"))
            if name not in seen_repos:
                seen_repos.add(name)
                src = parent if parent else repo
                w_repo.writerow([name, src.get("language") or "N/A", src.get("stargazers_count", 0), ""])
        if len(data) < 100: break
    return results

def _crawl_watches(u, seen_repos, w_repo):
    """Returns list of (user, repo, timestamp, 'watch') — subscriptions/watched repos."""
    results = []
    for page in range(1, MAX_STARRED_PAGES + 1):
        data = _get_rest(f"{BASE_URL}/users/{u}/subscriptions", {"per_page": 100, "page": page})
        if not data or not isinstance(data, list): break
        for repo in data:
            name = repo.get("full_name")
            if not name: continue
            ts_str = repo.get("updated_at", "")
            try:
                ts = int(datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").timestamp())
            except Exception:
                ts = int(time.time())
            results.append((u, name, ts, "watch"))
            if name not in seen_repos:
                seen_repos.add(name)
                w_repo.writerow([name, repo.get("language") or "N/A", repo.get("stargazers_count", 0), ""])
        if len(data) < 100: break
    return results

def crawl(users, cp: dict):
    start_idx = cp.get("crawl_idx", 0)
    total_int = cp.get("total_interactions", 0)
    if start_idx >= len(users): return RAW_CSV, RAW_REPO_CSV

    signals = []
    if COLLECT_STARS:   signals.append("star")
    if COLLECT_FORKS:   signals.append("fork")
    if COLLECT_WATCHES: signals.append("watch")
    logger.info(f"--- PHASE 2: Crawl [{', '.join(signals)}] (Mục tiêu: {CRAWL_INT_TARGET:,}) ---")

    seen_repos = set()
    mode = "a" if start_idx > 0 else "w"

    with open(RAW_CSV, mode, newline="", encoding="utf-8") as f_int, \
         open(RAW_REPO_CSV, mode, newline="", encoding="utf-8") as f_repo:
        w_int, w_repo = csv.writer(f_int), csv.writer(f_repo)
        if start_idx == 0:
            w_int.writerow(["user", "repo", "timestamp", "interaction_type"])
            w_repo.writerow(["repo", "language", "stars", "description"])

        for idx in range(start_idx, len(users)):
            u = users[idx]
            user_ints = []

            if COLLECT_STARS:
                user_ints.extend(_crawl_starred(u, seen_repos, w_repo))
            if COLLECT_FORKS:
                user_ints.extend(_crawl_forks(u, seen_repos, w_repo))
            if COLLECT_WATCHES:
                user_ints.extend(_crawl_watches(u, seen_repos, w_repo))

            if user_ints:
                w_int.writerows(user_ints)
                total_int += len(user_ints)

            if idx > 0 and idx % 20 == 0:
                logger.info(f"User {idx}/{len(users)} | Tổng tương tác: {total_int:,}")

            if idx > 0 and idx % 200 == 0:
                save_cp({"phase": "crawl", "users": users, "crawl_idx": idx + 1, "total_interactions": total_int})

            if total_int >= CRAWL_INT_TARGET:
                logger.info("[!] Đã đạt mục tiêu crawl thô.")
                break

    save_cp({"phase": "process", "users": users, "total_interactions": total_int})
    return RAW_CSV, RAW_REPO_CSV

# ─────────────────────────────────────────────
# PHASE 3: K-core
# ─────────────────────────────────────────────
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
            itype = r.get("interaction_type", "star")
            weight = INTERACTION_WEIGHTS.get(itype, 1.0)
            # Store weighted timestamp so stronger signals rank higher
            data[r["user"]][r["repo"]] = float(r["timestamp"]) * weight
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

#==============================================================================================================
def get_followers(username):
    url = f"{BASE_URL}/users/{username}/followers"
    params = {"per_page": 50, "page": 1}
    try:
        res = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.star+json"}, params=params, timeout=10)
        if res.status_code == 200:
            return [f["login"] for f in res.json()]
    except Exception:
        pass
    return []

def get_starred_repos(username):
    url = f"{BASE_URL}/users/{username}/starred"
    params = {"per_page": 100, "page": 1}
    try:
        res = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.star+json"}, params=params, timeout=10)
        if res.status_code == 200:
            return [("star", item) for item in res.json()]
        elif res.status_code == 403:
            logger.warning("Chạm trần API (Rate Limit), tạm nghỉ 60s...")
            time.sleep(60)
            return get_starred_repos(username)
    except Exception as e:
        pass
    return []

def get_forked_repos(username):
    url = f"{BASE_URL}/users/{username}/repos"
    params = {"per_page": 100, "page": 1, "type": "forks"}
    try:
        res = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}, params=params, timeout=10)
        if res.status_code == 200:
            return [("fork", r) for r in res.json() if r.get("fork")]
    except Exception:
        pass
    return []

def get_watched_repos(username):
    url = f"{BASE_URL}/users/{username}/subscriptions"
    params = {"per_page": 100, "page": 1}
    try:
        res = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}, params=params, timeout=10)
        if res.status_code == 200:
            return [("watch", r) for r in res.json()]
    except Exception:
        pass
    return []

def run_crawl_more():
    RAW_INT_CSV = "data/interactions.csv"
    RAW_REPO_CSV = "data/repos.csv"
    TARGET_NEW_USERS = 4000
    
    logger.info("=" * 60)
    logger.info(f"chuẩn bị cào thêm {TARGET_NEW_USERS} user")
    logger.info("=" * 60)

    existing_users = set()
    seed_users = []
    if os.path.exists(RAW_INT_CSV):
        with open(RAW_INT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_users.add(row["user"])
                seed_users.append(row["user"])

    existing_repos = set()
    if os.path.exists(RAW_REPO_CSV):
        with open(RAW_REPO_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_repos.add(row["repo"])

    logger.info(f"Đã load {len(existing_users):,} users và {len(existing_repos):,} repos từ data cũ.")
    logger.debug(f"Các user đã thăm: {existing_users}") # Dùng debug cho danh sách dài để không rác log

    new_users = []
    queue = deque(seed_users[-1000:])
    visited = set(existing_users)

    logger.info(f"Đang rò tìm {TARGET_NEW_USERS} user mới (BFS)...")
    while queue and len(new_users) < TARGET_NEW_USERS:
        current_u = queue.popleft()
        followers = get_followers(current_u)

        for f in followers:
            if f not in visited:
                visited.add(f)
                new_users.append(f)
                queue.append(f)
                if len(new_users) >= TARGET_NEW_USERS:
                    break
        time.sleep(0.5)

    logger.info(f"Đã đạt mục tiêu {len(new_users)} users mới. Bắt đầu thu thập data!")

    new_int_count = 0
    with open(RAW_INT_CSV, "a", newline="", encoding="utf-8") as f_int, \
        open(RAW_REPO_CSV, "a", newline="", encoding="utf-8") as f_repo:

        w_int = csv.writer(f_int)
        w_repo = csv.writer(f_repo)

        for i, u in enumerate(new_users):
            all_items = []
            if COLLECT_STARS:   all_items.extend(get_starred_repos(u))
            if COLLECT_FORKS:   all_items.extend(get_forked_repos(u))
            if COLLECT_WATCHES: all_items.extend(get_watched_repos(u))

            if not all_items:
                continue

            for itype, item in all_items:
                if itype == "star":
                    repo_name = item.get("repo", {}).get("full_name") or item.get("full_name")
                    lang = item.get("repo", {}).get("language") or item.get("language", "N/A")
                    st   = item.get("repo", {}).get("stargazers_count") or item.get("stargazers_count", 0)
                    desc = str(item.get("repo", {}).get("description") or item.get("description", ""))[:100]
                elif itype == "fork":
                    parent = item.get("parent") or {}
                    repo_name = parent.get("full_name") or item.get("full_name")
                    lang = parent.get("language") or item.get("language", "N/A")
                    st   = parent.get("stargazers_count") or item.get("stargazers_count", 0)
                    desc = str(parent.get("description") or item.get("description", ""))[:100]
                else:  # watch
                    repo_name = item.get("full_name")
                    lang = item.get("language", "N/A")
                    st   = item.get("stargazers_count", 0)
                    desc = str(item.get("description", ""))[:100]

                if not repo_name:
                    continue

                timestamp = int(time.time())
                w_int.writerow([u, repo_name, timestamp, itype])
                new_int_count += 1

                if repo_name not in existing_repos:
                    existing_repos.add(repo_name)
                    w_repo.writerow([repo_name, lang or "N/A", st, desc.replace("\n", "")])

            if (i + 1) % 50 == 0:
                logger.info(f"[Tiến độ: {i + 1}/{TARGET_NEW_USERS}] Cào thêm được {new_int_count:,} tương tác...")
                f_int.flush()
                f_repo.flush()

    logger.info(f"hoàn tất ! Đã cào thêm {TARGET_NEW_USERS} users và {new_int_count:,} interactions.")
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
def main():
    logger.info("="*50)
    logger.info("GITHUB CRAWLER V3 - TARGET 1.2M")
    logger.info("="*50)
    
    cp = load_cp()
    users = collect_users(cp)
    raw_int, raw_repo = crawl(users, load_cp())

    data = kcore(load_raw(raw_int))
    user2id, item2id, remapped = remap(data)
    train, test = split(remapped)

    export_all(train, test, user2id, item2id, remapped, raw_repo)
    logger.info("Hoành thành pipeline")

if __name__ == "__main__":
    main()

  
    # raw_interaction = os.path.join("data", "interactions.csv")
    # raw_repo = os.path.join("data", "repos.csv")
    # if not os.path.exists(raw_interaction) or not os.path.exists(raw_repo):
    #     logger.error("Không tìm thấy file CSV! Hãy kiểm tra lại đường dẫn.")

    # interactions_df = pd.read_csv(raw_interaction)
    # repos_df = pd.read_csv(raw_repo)
    # df_clean   = clean(interactions_df, repos_df, min_stars=10)
    # data  = build_interaction_dict(df_clean)
    
    # data = kcore(data )
    # user2id, item2id, remapped = remap(data)
    # train, test = split(remapped)

    # export_all(train, test, user2id, item2id, remapped, raw_repo)
    # print_stats(user2id, item2id, remapped, train, test)

    # save_cp({"phase": "done"})
    # logger.info("PIPELINE DONE")