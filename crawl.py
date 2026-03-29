import os, csv, time, json, requests, logging
from datetime import datetime
from collections import defaultdict, deque
import pandas as pd
from dotenv import load_dotenv
# ─────────────────────────────────────────────
# CẤU HÌNH LOGGING
# ─────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("crawler.log", encoding="utf-8"), 
        logging.StreamHandler()                              
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG NÂNG CẤP (Mục tiêu 1.2M+ Interactions)
# ─────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("PAS_CRAWL")
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

def _get_rest(url, params=None):
    for attempt in range(5):
        try:
            r = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.star+json"}, params=params, timeout=15)
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
# PHASE 2: Crawl stars (Deep Star)
# ─────────────────────────────────────────────
def crawl(users, cp: dict):
    start_idx = cp.get("crawl_idx", 0)
    total_int = cp.get("total_interactions", 0)
    if start_idx >= len(users): return RAW_CSV, RAW_REPO_CSV

    logger.info(f"--- PHASE 2: Crawl Star (Mục tiêu: {CRAWL_INT_TARGET:,}) ---")
    seen_repos = set()
    mode = "a" if start_idx > 0 else "w"

    with open(RAW_CSV, mode, newline="", encoding="utf-8") as f_int, \
         open(RAW_REPO_CSV, mode, newline="", encoding="utf-8") as f_repo:
        w_int, w_repo = csv.writer(f_int), csv.writer(f_repo)
        if start_idx == 0:
            w_int.writerow(["user", "repo", "timestamp"])
            w_repo.writerow(["repo", "language", "stars", "description"])

        for idx in range(start_idx, len(users)):
            u = users[idx]
            user_ints = []

            for page in range(1, MAX_STARRED_PAGES + 1):
                data = _get_rest(f"{BASE_URL}/users/{u}/starred", {"per_page": 100, "page": page})
                if not data or not isinstance(data, list): break

                for item in data:
                    repo = item["repo"]
                    name = repo.get("full_name")
                    ts = int(datetime.strptime(item["starred_at"], "%Y-%m-%dT%H:%M:%SZ").timestamp())
                    user_ints.append((u, name, ts))
                    if name not in seen_repos:
                        seen_repos.add(name)
                        w_repo.writerow([name, repo.get("language") or "N/A", repo.get("stargazers_count", 0), ""])
                if len(data) < 100: break

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
            return res.json()
        elif res.status_code == 403:
            logger.warning("Chạm trần API (Rate Limit), tạm nghỉ 60s...")
            time.sleep(60)
            return get_starred_repos(username)
    except Exception as e:
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
    logger.debug(f"Các user đã thăm: {existing_users}") 

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
            stars = get_starred_repos(u)
            if not stars:
                continue

           

            for item in stars:
                repo_data = item.get("repo", item) 
                
                repo_name = repo_data.get("full_name")
                if not repo_name:
                    continue

                timestamp = int(time.time())
                w_int.writerow([u, repo_name, timestamp])
                new_int_count += 1

                if repo_name not in existing_repos:
                    existing_repos.add(repo_name)
                    
                    lang = repo_data.get("language") or "N/A"
                    st = repo_data.get("stargazers_count", 0)
                    
                    w_repo.writerow([repo_name, lang, st, ""])

            if (i + 1) % 50 == 0:
                logger.info(f"[Tiến độ: {i + 1}/{TARGET_NEW_USERS}] Cào thêm được {new_int_count:,} tương tác...")
                f_int.flush()
                f_repo.flush()

    logger.info(f"hoàn tất ! Đã cào thêm {TARGET_NEW_USERS} users và {new_int_count:,} interactions.")

def main():
    logger.info("="*50)
    logger.info("GITHUB CRAWLER V3 - TARGET 1.2M")
    logger.info("="*50)
    
    cp = load_cp()
    users = collect_users(cp)
    raw_int, raw_repo = crawl(users, load_cp())


if __name__ == "__main__":
  
    choice = 1
    if choice == 1:
        main()
    else: 
        run_crawl_more()
    
    