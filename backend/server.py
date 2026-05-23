from fastapi import FastAPI, APIRouter, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import httpx
import secrets
import json
from urllib.parse import quote_plus
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cinediario")
security_logger = logging.getLogger("cinediario.security")

# ===== MongoDB connection (MongoDB Atlas) =====
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(
    mongo_url,
    serverSelectionTimeoutMS=8000,
    maxPoolSize=50,
    retryWrites=True,
)
db = client[os.environ['DB_NAME']]

# ===== TMDB / Gemini config =====
TMDB_TOKEN = os.environ['TMDB_TOKEN']
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
TMDB_PROFILE_IMG = "https://image.tmdb.org/t/p/w185"
TMDB_HEADERS = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
TMDB_LANG = "it-IT"

# ===== Security config (da env) =====
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get('ALLOWED_ORIGINS', '').split(',') if o.strip()]
ALLOWED_HOSTS = [h.strip() for h in os.environ.get('ALLOWED_HOSTS', '*').split(',') if h.strip()]
MAX_BODY_BYTES = int(os.environ.get('MAX_BODY_BYTES', 1_048_576))
FIREBASE_PROJECT_ID = os.environ['FIREBASE_PROJECT_ID']
APP_ENV = os.environ.get('APP_ENV', 'production').lower()
IS_PROD = APP_ENV == 'production'

# ===== Firebase Admin initialization =====
def _init_firebase():
    if firebase_admin._apps:
        return
    project_id = FIREBASE_PROJECT_ID
    client_email = os.environ['FIREBASE_CLIENT_EMAIL']
    private_key = os.environ['FIREBASE_PRIVATE_KEY'].replace('\\n', '\n')
    cred = credentials.Certificate({
        "type": "service_account",
        "project_id": project_id,
        "private_key": private_key,
        "client_email": client_email,
        "token_uri": "https://oauth2.googleapis.com/token",
    })
    firebase_admin.initialize_app(cred, {"projectId": project_id})

_init_firebase()

# ===== Rate limiter avanzato =====
def _rate_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and len(auth) > 20:
        return f"bearer:{auth[7:27]}"
    return get_remote_address(request)

limiter = Limiter(key_func=_rate_key, default_limits=["120/minute"])

app = FastAPI(
    title="CineDiario API",
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
    openapi_url=None if IS_PROD else "/openapi.json",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ===== Security middlewares =====
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        try:
            del response.headers["Server"]
        except KeyError:
            pass
        return response

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "Payload troppo grande"},
            )
        return await call_next(request)

api_router = APIRouter(prefix="/api")

# ===== Models =====
class UserMovieReq(BaseModel):
    tmdb_id: int
    status: str
    rating: Optional[float] = None
    notes: Optional[str] = None
    rating_directing: Optional[float] = None
    rating_acting: Optional[float] = None
    rating_screenplay: Optional[float] = None
    rating_soundtrack: Optional[float] = None
    rating_cinematography: Optional[float] = None

class UpdateMovieReq(BaseModel):
    status: Optional[str] = None
    rating: Optional[float] = None
    notes: Optional[str] = None
    rating_directing: Optional[float] = None
    rating_acting: Optional[float] = None
    rating_screenplay: Optional[float] = None
    rating_soundtrack: Optional[float] = None
    rating_cinematography: Optional[float] = None

# ===== Helpers =====
def serialize_user(user: dict) -> dict:
    return {
        "user_id": user["user_id"],
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "picture": user.get("picture"),
        "is_guest": False,
        "friend_code": user.get("friend_code"),
    }

def _generate_friend_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    suffix = "".join(secrets.choice(alphabet) for _ in range(6))
    return f"CIN-{suffix}"

async def ensure_friend_code(user: dict) -> str:
    code = user.get("friend_code")
    if code:
        return code
    for _ in range(10):
        candidate = _generate_friend_code()
        existing = await db.users.find_one({"friend_code": candidate}, {"_id": 1})
        if not existing:
            await db.users.update_one(
                {"user_id": user["user_id"]},
                {"$set": {"friend_code": candidate}},
            )
            user["friend_code"] = candidate
            return candidate
    fallback = f"CIN-{uuid.uuid4().hex[:6].upper()}"
    await db.users.update_one({"user_id": user["user_id"]}, {"$set": {"friend_code": fallback}})
    user["friend_code"] = fallback
    return fallback

def _extract_bearer_token(authorization: Optional[str], client_ip: str) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        security_logger.warning(f"AUTH_FAIL missing_token ip={client_ip}")
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization[7:].strip()
    if not token or len(token) < 10:
        security_logger.warning(f"AUTH_FAIL malformed_token ip={client_ip}")
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

def _verify_firebase_token(id_token: str, client_ip: str) -> dict:
    expected_iss = f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}"
    if "." in id_token and len(id_token) > 100:
        try:
            decoded = firebase_auth.verify_id_token(id_token, check_revoked=True)
            if decoded.get("aud") == FIREBASE_PROJECT_ID and decoded.get("iss") == expected_iss:
                return decoded
            security_logger.warning(f"AUTH_FAIL wrong_project ip={client_ip} aud={decoded.get('aud')}")
            raise HTTPException(status_code=401, detail="Token not for this project")
        except firebase_auth.RevokedIdTokenError:
            security_logger.warning(f"AUTH_FAIL revoked_token ip={client_ip}")
            raise HTTPException(status_code=401, detail="Token revoked")
        except firebase_auth.ExpiredIdTokenError:
            raise HTTPException(status_code=401, detail="Token expired")
        except firebase_auth.UserDisabledError:
            security_logger.warning(f"AUTH_FAIL user_disabled ip={client_ip}")
            raise HTTPException(status_code=401, detail="User disabled")
        except HTTPException:
            raise
        except firebase_auth.InvalidIdTokenError:
            pass 
        except Exception as e:
            security_logger.error(f"AUTH_FAIL verify_error ip={client_ip} err={e}")

    if 10 <= len(id_token) <= 200 and all(c.isalnum() or c in "-_" for c in id_token):
        try:
            ur = firebase_auth.get_user(id_token)
            return {
                "uid": ur.uid,
                "email": (ur.email or "").lower(),
                "email_verified": bool(ur.email_verified),
                "name": ur.display_name or "",
                "picture": ur.photo_url,
                "aud": FIREBASE_PROJECT_ID,
                "iss": expected_iss,
                "firebase": {"sign_in_provider": "uid_compat"},
            }
        except firebase_auth.UserNotFoundError:
            security_logger.warning(f"AUTH_FAIL unknown_uid ip={client_ip} uid_prefix={id_token[:8]}")
        except Exception as e:
            security_logger.error(f"AUTH_FAIL uid_lookup_error ip={client_ip} err={e}")

    security_logger.warning(f"AUTH_FAIL invalid_token ip={client_ip} len={len(id_token)} prefix={id_token[:20]}")
    raise HTTPException(status_code=401, detail="Invalid Firebase token")

def _build_user_profile(decoded: dict) -> dict:
    uid = decoded.get("uid") or decoded.get("user_id")
    if not uid:
        raise HTTPException(status_code=401, detail="UID mancante nel token")
    email = (decoded.get("email") or "").lower()
    return {
        "uid": uid,
        "email": email,
        "email_verified": bool(decoded.get("email_verified")),
        "name": decoded.get("name") or (email.split("@")[0] if email else "Utente"),
        "picture": decoded.get("picture"),
        "provider_id": decoded.get("firebase", {}).get("sign_in_provider"),
    }

async def _upsert_user(profile: dict) -> dict:
    uid = profile["uid"]
    now = datetime.now(timezone.utc)
    user = await db.users.find_one({"user_id": uid}, {"_id": 0})
    if not user:
        user = {
            "user_id": uid,
            "email": profile["email"],
            "email_verified": profile["email_verified"],
            "name": profile["name"],
            "picture": profile["picture"],
            "auth_provider": "firebase",
            "auth_provider_id": profile["provider_id"],
            "created_at": now,
            "last_login_at": now,
        }
        try:
            await db.users.insert_one(user)
            security_logger.info(f"AUTH_NEW_USER uid={uid} provider={profile['provider_id']}")
        except Exception:
            user = await db.users.find_one({"user_id": uid}, {"_id": 0}) or user
    else:
        updates = {"last_login_at": now}
        for key in ("name", "picture", "email", "email_verified"):
            if profile[key] and user.get(key) != profile[key]:
                updates[key] = profile[key]
        await db.users.update_one({"user_id": uid}, {"$set": updates})
        user.update(updates)
    await ensure_friend_code(user)
    return user

async def get_current_user(request: Request, authorization: Optional[str] = Header(None)) -> dict:
    client_ip = get_remote_address(request)
    token = _extract_bearer_token(authorization, client_ip)
    decoded = _verify_firebase_token(token, client_ip)
    profile = _build_user_profile(decoded)
    return await _upsert_user(profile)

async def tmdb_get(path: str, params: dict = None) -> dict:
    p = {"language": TMDB_LANG}
    if params:
        p.update(params)
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get(f"{TMDB_BASE}{path}", headers=TMDB_HEADERS, params=p)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"TMDB error: {r.status_code}")
        return r.json()

def fmt_movie(m: dict) -> dict:
    return {
        "tmdb_id": m["id"],
        "title": m.get("title") or m.get("original_title", ""),
        "overview": m.get("overview"),
        "poster_url": f"{TMDB_IMG}{m['poster_path']}" if m.get("poster_path") else None,
        "backdrop_url": f"{TMDB_IMG}{m['backdrop_path']}" if m.get("backdrop_path") else None,
        "release_date": m.get("release_date"),
        "vote_average": m.get("vote_average", 0),
    }

def pick_trailer(videos: dict) -> Optional[str]:
    results = videos.get("results", []) if videos else []
    def score(v):
        if v.get("site") != "YouTube":
            return -1
        s = 0
        if v.get("type") == "Trailer":
            s += 10
        elif v.get("type") == "Teaser":
            s += 5
        if v.get("official"):
            s += 3
        return s
    youtube = [v for v in results if v.get("site") == "YouTube" and v.get("key")]
    if not youtube:
        return None
    best = max(youtube, key=score)
    return f"https://www.youtube.com/watch?v={best['key']}"

def compute_overall(req: dict) -> Optional[float]:
    if req.get("rating") is not None:
        return req.get("rating")
    breakdown = [
        req.get("rating_directing"),
        req.get("rating_acting"),
        req.get("rating_screenplay"),
        req.get("rating_soundtrack"),
        req.get("rating_cinematography"),
    ]
    filled = [r for r in breakdown if r is not None]
    if not filled:
        return None
    avg = round(sum(filled) / len(filled), 1)
    return avg

PROVIDER_URLS = {
    8:   "https://www.netflix.com/search?q={q}",
    9:   "https://www.primevideo.com/region/eu/search/ref=atv_nb_sr?phrase={q}",
    119: "https://www.primevideo.com/region/eu/search/ref=atv_nb_sr?phrase={q}",
    337: "https://www.disneyplus.com/it-it/search?q={q}",
    350: "https://tv.apple.com/it/search?term={q}",
    2:   "https://tv.apple.com/it/search?term={q}",
    217: "https://www.raiplay.it/ricerca.html?q={q}",
    228: "https://mediasetinfinity.mediaset.it/ricerca/{q}",
    39:  "https://www.nowtv.it/cerca?q={q}",
    531: "https://www.paramountplus.com/it/search/?query={q}",
    283: "https://www.crunchyroll.com/it/search?q={q}",
    192: "https://www.youtube.com/results?search_query={q}+film",
    3:   "https://play.google.com/store/search?q={q}&c=movies",
    68:  "https://www.microsoft.com/it-it/search?q={q}",
    11:  "https://www.mubi.com/it/search/films?query={q}",
    100: "https://www.guidatv.sky.it/cerca?q={q}",
    1796: "https://www.netflix.com/search?q={q}",
    1899: "https://play.max.com/search?q={q}",
}

def build_provider_url(provider_id: int, title: str, fallback_link: Optional[str]) -> str:
    template = PROVIDER_URLS.get(provider_id)
    if template:
        return template.format(q=quote_plus(title))
    return fallback_link or f"https://www.google.com/search?q={quote_plus(title)}+streaming"

async def fetch_watch_providers(tmdb_id: int, title: str) -> list:
    try:
        data = await tmdb_get(f"/movie/{tmdb_id}/watch/providers")
    except Exception:
        return []
    it = (data.get("results") or {}).get("IT") or {}
    fallback_link = it.get("link")
    out = []
    seen_ids = set()
    cats = [
        ("flatrate", "stream", "Streaming"),
        ("free", "stream", "Gratis"),
        ("ads", "stream", "Con pubblicità"),
        ("rent", "rent", "Noleggio"),
        ("buy", "buy", "Acquisto"),
    ]
    for tmdb_key, kind, label in cats:
        for p in it.get(tmdb_key, []) or []:
            pid = p.get("provider_id")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            out.append({
                "provider_id": pid,
                "name": p.get("provider_name"),
                "logo_url": f"{TMDB_PROFILE_IMG}{p['logo_path']}" if p.get("logo_path") else None,
                "kind": kind,
                "kind_label": label,
                "url": build_provider_url(pid, title, fallback_link),
            })
    return out

# ===== Auth Routes =====
@api_router.get("/auth/me")
@limiter.limit("30/minute")
async def auth_me(request: Request, user: dict = Depends(get_current_user)):
    return {"user": serialize_user(user)}

# ===== USER MOVIES =====
VALID_STATUSES = {"watched", "watchlist", "favorite", "watching"}

async def _movie_snapshot(tmdb_id: int) -> dict:
    try:
        details = await tmdb_get(f"/movie/{tmdb_id}")
        return {
            "tmdb_id": details["id"],
            "title": details.get("title") or details.get("original_title", ""),
            "poster_url": f"{TMDB_IMG}{details['poster_path']}" if details.get("poster_path") else None,
            "backdrop_url": f"{TMDB_IMG}{details['backdrop_path']}" if details.get("backdrop_path") else None,
            "release_date": details.get("release_date"),
            "overview": details.get("overview"),
            "vote_average": details.get("vote_average", 0),
            "genres": [g.get("name") for g in details.get("genres", []) if g.get("name")],
            "runtime": details.get("runtime"),
        }
    except Exception:
        return {"tmdb_id": tmdb_id, "title": "", "poster_url": None}

def _serialize_user_movie(doc: dict) -> dict:
    # IL FIX E' QUI: Estrarre locandina e titolo dalla sottocartella movie per farle leggere all'app!
    movie_obj = doc.get("movie") or {}
    return {
        "tmdb_id": doc["tmdb_id"],
        "status": doc.get("status"),
        "rating": doc.get("rating"),
        "overall": doc.get("overall"),
        "title": movie_obj.get("title", ""),
        "poster_url": movie_obj.get("poster_url"),
        "backdrop_url": movie_obj.get("backdrop_url"),
        "notes": doc.get("notes"),
        "rating_directing": doc.get("rating_directing"),
        "rating_acting": doc.get("rating_acting"),
        "rating_screenplay": doc.get("rating_screenplay"),
        "rating_soundtrack": doc.get("rating_soundtrack"),
        "rating_cinematography": doc.get("rating_cinematography"),
        "movie": movie_obj,
        "created_at": (doc.get("created_at") or datetime.now(timezone.utc)).isoformat(),
        "updated_at": (doc.get("updated_at") or datetime.now(timezone.utc)).isoformat(),
    }

@api_router.get("/user/movies")
@limiter.limit("60/minute")
async def list_user_movies(request: Request, status: Optional[str] = None, user: dict = Depends(get_current_user)):
    query = {"user_id": user["user_id"]}
    if status:
        if status not in VALID_STATUSES:
            raise HTTPException(status_code=400, detail="status non valido")
        query["status"] = status
    cursor = db.user_movies.find(query, {"_id": 0}).sort("updated_at", -1).limit(500)
    return [_serialize_user_movie(d) async for d in cursor]

@api_router.get("/user/movies/{tmdb_id}")
@limiter.limit("60/minute")
async def get_user_movie(request: Request, tmdb_id: int, user: dict = Depends(get_current_user)):
    doc = await db.user_movies.find_one({"user_id": user["user_id"], "tmdb_id": tmdb_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Film non presente nel diario")
    return _serialize_user_movie(doc)

@api_router.post("/user/movies")
@limiter.limit("30/minute")
async def upsert_user_movie(request: Request, req: UserMovieReq, user: dict = Depends(get_current_user)):
    if req.status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail="status non valido")
    snapshot = await _movie_snapshot(req.tmdb_id)
    now = datetime.now(timezone.utc)
    data = req.model_dump()
    data["overall"] = compute_overall(data)
    update = {
        "user_id": user["user_id"],
        "tmdb_id": req.tmdb_id,
        "status": req.status,
        "rating": req.rating,
        "overall": data["overall"],
        "notes": req.notes,
        "rating_directing": req.rating_directing,
        "rating_acting": req.rating_acting,
        "rating_screenplay": req.rating_screenplay,
        "rating_soundtrack": req.rating_soundtrack,
        "rating_cinematography": req.rating_cinematography,
        "movie": snapshot,
        "updated_at": now,
    }
    await db.user_movies.update_one(
        {"user_id": user["user_id"], "tmdb_id": req.tmdb_id},
        {"$set": update, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    doc = await db.user_movies.find_one({"user_id": user["user_id"], "tmdb_id": req.tmdb_id}, {"_id": 0})
    return _serialize_user_movie(doc)

@api_router.patch("/user/movies/{tmdb_id}")
@limiter.limit("30/minute")
async def update_user_movie(request: Request, tmdb_id: int, req: UpdateMovieReq, user: dict = Depends(get_current_user)):
    existing = await db.user_movies.find_one({"user_id": user["user_id"], "tmdb_id": tmdb_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Film non presente nel diario")
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if "status" in updates and updates["status"] not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail="status non valido")
    merged = {**existing, **updates}
    updates["overall"] = compute_overall(merged)
    updates["updated_at"] = datetime.now(timezone.utc)
    await db.user_movies.update_one({"user_id": user["user_id"], "tmdb_id": tmdb_id}, {"$set": updates})
    doc = await db.user_movies.find_one({"user_id": user["user_id"], "tmdb_id": tmdb_id}, {"_id": 0})
    return _serialize_user_movie(doc)

@api_router.delete("/user/movies/{tmdb_id}")
@limiter.limit("30/minute")
async def delete_user_movie(request: Request, tmdb_id: int, user: dict = Depends(get_current_user)):
    result = await db.user_movies.delete_one({"user_id": user["user_id"], "tmdb_id": tmdb_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Film non presente nel diario")
    return {"ok": True}

@api_router.get("/user/stats")
@limiter.limit("30/minute")
async def user_stats(request: Request, user: dict = Depends(get_current_user)):
    pipeline = [
        {"$match": {"user_id": user["user_id"]}},
        {"$group": {
            "_id": "$status",
            "count": {"$sum": 1},
            "avg_rating": {"$avg": "$overall"},
        }},
    ]
    by_status = {}
    total = 0
    sum_minutes = 0
    async for row in db.user_movies.aggregate(pipeline):
        by_status[row["_id"] or "unknown"] = {
            "count": row["count"],
            "avg_rating": round(row["avg_rating"], 2) if row["avg_rating"] else None,
        }
        total += row["count"]
    runtime_cursor = db.user_movies.find(
        {"user_id": user["user_id"], "status": "watched"},
        {"_id": 0, "movie.runtime": 1},
    )
    async for d in runtime_cursor:
        rt = ((d.get("movie") or {}).get("runtime")) or 0
        sum_minutes += rt
        
    # IL FIX E' QUI: Modificato l'output per combaciare esattamente con la richiesta dell'app
    return {
        "total_watched": by_status.get("watched", {}).get("count", 0),
        "total_watchlist": by_status.get("watchlist", {}).get("count", 0),
        "average_rating": by_status.get("watched", {}).get("avg_rating"),
        "total": total,
        "by_status": by_status,
        "watched_minutes": sum_minutes,
        "watched_hours": round(sum_minutes / 60, 1),
    }

@api_router.get("/user/recommendations")
@limiter.limit("20/minute")
async def user_recommendations(request: Request, user: dict = Depends(get_current_user)):
    cursor = db.user_movies.find(
        {"user_id": user["user_id"], "status": "watched", "overall": {"$gte": 4}},
        {"_id": 0, "tmdb_id": 1},
    ).sort("updated_at", -1).limit(3)
    seeds = [d["tmdb_id"] async for d in cursor]

    already_ids = set()
    async for d in db.user_movies.find({"user_id": user["user_id"]}, {"_id": 0, "tmdb_id": 1}):
        already_ids.add(d["tmdb_id"])

    if not seeds:
        data = await tmdb_get("/movie/popular", {"page": 1})
        recs = [fmt_movie(m) for m in data.get("results", []) if m.get("id") not in already_ids]
        return {"recommendations": recs[:20], "message": "Vota qualche film per ricevere consigli personalizzati!"}

    out = []
    seen = set()
    for seed in seeds:
        try:
            data = await tmdb_get(f"/movie/{seed}/recommendations", {"page": 1})
        except Exception:
            continue
        for m in data.get("results", []):
            mid = m.get("id")
            if mid and mid not in seen and mid not in already_ids:
                seen.add(mid)
                out.append(fmt_movie(m))
    return {"recommendations": out[:20]}

# ===== FRIENDS =====
def _friendship_key(a: str, b: str) -> dict:
    lo, hi = sorted([a, b])
    return {"user_lo": lo, "user_hi": hi}

def _other_user(fr: dict, me: str) -> str:
    return fr["user_hi"] if fr["user_lo"] == me else fr["user_lo"]

async def _friend_profile(user_id: str) -> dict:
    other = await db.users.find_one(
        {"user_id": user_id},
        {"_id": 0, "user_id": 1, "name": 1, "picture": 1, "friend_code": 1},
    )
    return other or {"user_id": user_id, "name": "Utente", "picture": None, "friend_code": None}

@api_router.get("/friends")
@limiter.limit("60/minute")
async def list_friends(request: Request, user: dict = Depends(get_current_user)):
    me = user["user_id"]
    cursor = db.friendships.find(
        {"$or": [{"user_lo": me}, {"user_hi": me}], "status": "accepted"},
        {"_id": 0},
    )
    out = []
    async for fr in cursor:
        out.append(await _friend_profile(_other_user(fr, me)))
    return out

@api_router.get("/friends/requests")
@limiter.limit("60/minute")
async def list_friend_requests(request: Request, user: dict = Depends(get_current_user)):
    me = user["user_id"]
    incoming, outgoing = [], []
    cursor = db.friendships.find(
        {"$or": [{"user_lo": me}, {"user_hi": me}], "status": "pending"},
        {"_id": 0},
    )
    async for fr in cursor:
        other_profile = await _friend_profile(_other_user(fr, me))
        entry = {
            "id": fr.get("request_id"),
            "user": other_profile,
            "created_at": (fr.get("created_at") or datetime.now(timezone.utc)).isoformat(),
        }
        if fr.get("requested_by") == me:
            outgoing.append(entry)
        else:
            incoming.append(entry)
    return {"incoming": incoming, "outgoing": outgoing}

class FriendRequestReq(BaseModel):
    friend_code: str

@api_router.post("/friends/requests")
@limiter.limit("10/minute")
async def send_friend_request(request: Request, req: FriendRequestReq, user: dict = Depends(get_current_user)):
    code = req.friend_code.strip().upper()
    if not code:
        raise HTTPException(status_code=400, detail="Codice amico mancante")
    target = await db.users.find_one({"friend_code": code}, {"_id": 0})
    if not target:
        raise HTTPException(status_code=404, detail="Codice amico non trovato")
    if target["user_id"] == user["user_id"]:
        raise HTTPException(status_code=400, detail="Non puoi aggiungere te stesso")
    key = _friendship_key(user["user_id"], target["user_id"])
    existing = await db.friendships.find_one(key)
    if existing:
        if existing.get("status") == "accepted":
            raise HTTPException(status_code=400, detail="Siete già amici")
        raise HTTPException(status_code=400, detail="Richiesta già in corso")
    request_id = f"fr_{uuid.uuid4().hex[:12]}"
    await db.friendships.insert_one({
        **key,
        "request_id": request_id,
        "requested_by": user["user_id"],
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
    })
    return {"ok": True, "request_id": request_id, "to": await _friend_profile(target["user_id"])}

@api_router.post("/friends/requests/{request_id}/accept")
@limiter.limit("30/minute")
async def accept_friend_request(request: Request, request_id: str, user: dict = Depends(get_current_user)):
    fr = await db.friendships.find_one({"request_id": request_id, "status": "pending"})
    if not fr:
        raise HTTPException(status_code=404, detail="Richiesta non trovata")
    me = user["user_id"]
    if me not in (fr["user_lo"], fr["user_hi"]) or fr.get("requested_by") == me:
        raise HTTPException(status_code=403, detail="Non puoi accettare questa richiesta")
    await db.friendships.update_one(
        {"request_id": request_id},
        {"$set": {"status": "accepted", "accepted_at": datetime.now(timezone.utc)}},
    )
    return {"ok": True, "friend": await _friend_profile(_other_user(fr, me))}

@api_router.post("/friends/requests/{request_id}/decline")
@limiter.limit("30/minute")
async def decline_friend_request(request: Request, request_id: str, user: dict = Depends(get_current_user)):
    fr = await db.friendships.find_one({"request_id": request_id, "status": "pending"})
    if not fr:
        raise HTTPException(status_code=404, detail="Richiesta non trovata")
    me = user["user_id"]
    if me not in (fr["user_lo"], fr["user_hi"]):
        raise HTTPException(status_code=403, detail="Non puoi rifiutare questa richiesta")
    await db.friendships.delete_one({"request_id": request_id})
    return {"ok": True}

@api_router.delete("/friends/{friend_user_id}")
@limiter.limit("30/minute")
async def remove_friend(request: Request, friend_user_id: str, user: dict = Depends(get_current_user)):
    key = _friendship_key(user["user_id"], friend_user_id)
    result = await db.friendships.delete_one({**key, "status": "accepted"})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Amicizia non trovata")
    return {"ok": True}

# ===== SHARES =====
class ShareReq(BaseModel):
    to_user_id: str
    tmdb_id: int
    message: Optional[str] = None

async def _are_friends(a: str, b: str) -> bool:
    key = _friendship_key(a, b)
    fr = await db.friendships.find_one({**key, "status": "accepted"})
    return fr is not None

@api_router.get("/shares/unread-count")
@limiter.limit("120/minute")
async def shares_unread_count(request: Request, user: dict = Depends(get_current_user)):
    count = await db.shares.count_documents({"to_user_id": user["user_id"], "read": False})
    return {"count": count}

@api_router.get("/shares/inbox")
@limiter.limit("60/minute")
async def shares_inbox(request: Request, user: dict = Depends(get_current_user)):
    cursor = db.shares.find({"to_user_id": user["user_id"]}, {"_id": 0}).sort("created_at", -1).limit(100)
    out = []
    async for s in cursor:
        out.append({
            "share_id": s["share_id"],
            "from": await _friend_profile(s["from_user_id"]),
            "tmdb_id": s["tmdb_id"],
            "movie": s.get("movie"),
            "message": s.get("message"),
            "read": bool(s.get("read", False)),
            "created_at": (s.get("created_at") or datetime.now(timezone.utc)).isoformat(),
        })
    return out

@api_router.post("/shares")
@limiter.limit("20/minute")
async def create_share(request: Request, req: ShareReq, user: dict = Depends(get_current_user)):
    if req.to_user_id == user["user_id"]:
        raise HTTPException(status_code=400, detail="Non puoi condividere con te stesso")
    if not await _are_friends(user["user_id"], req.to_user_id):
        raise HTTPException(status_code=403, detail="Puoi condividere solo con i tuoi amici")
    snapshot = await _movie_snapshot(req.tmdb_id)
    share_id = f"sh_{uuid.uuid4().hex[:12]}"
    await db.shares.insert_one({
        "share_id": share_id,
        "from_user_id": user["user_id"],
        "to_user_id": req.to_user_id,
        "tmdb_id": req.tmdb_id,
        "movie": snapshot,
        "message": (req.message or "").strip()[:500] or None,
        "read": False,
        "created_at": datetime.now(timezone.utc),
    })
    return {"ok": True, "share_id": share_id}

@api_router.post("/shares/{share_id}/read")
@limiter.limit("60/minute")
async def mark_share_read(request: Request, share_id: str, user: dict = Depends(get_current_user)):
    result = await db.shares.update_one(
        {"share_id": share_id, "to_user_id": user["user_id"]},
        {"$set": {"read": True, "read_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Condivisione non trovata")
    return {"ok": True}

# ===== AI MOVIE FINDER =====
class MovieFinderReq(BaseModel):
    answers: dict
    free_text: Optional[str] = None

# IL FIX E' QUI: Rotta rinominata per combaciare perfettamente con l'app
@api_router.post("/discover/ai-recommend")
@limiter.limit("15/minute")
async def ai_movie_finder(request: Request, req: MovieFinderReq, user: dict = Depends(get_current_user)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="LLM non configurato")
    answer_lines = []
    for k, v in (req.answers or {}).items():
        if v is None or v == "":
            continue
        answer_lines.append(f"- {k}: {v}")
    ans_block = "\n".join(answer_lines) if answer_lines else "(nessuna preferenza)"
    extra = (req.free_text or "").strip()[:500]

    prompt = f"""L'utente vuole consigli su 7 film da vedere. Ecco le sue risposte:
{ans_block}
Note libere: {extra or '(nessuna)'}

Restituisci SOLO un oggetto JSON con questo formato esatto:
{{"movies": [{{"title": "Titolo italiano del film", "year": 2010, "why": "Breve motivo (max 25 parole) in italiano"}}, ...]}}
Esattamente 7 film, in italiano, vari per stile ma coerenti con le risposte.
Non aggiungere testo prima o dopo il JSON, non usare markdown."""

    body = {
        "systemInstruction": {"parts": [{"text": "Sei un esperto cinefilo italiano. Consigli film esistenti reali, mai inventati. Rispondi solo JSON."}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.8, "responseMimeType": "application/json"},
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(GEMINI_URL, params={"key": GEMINI_API_KEY}, json=body)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="Impossibile generare consigli")
        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise HTTPException(status_code=502, detail="Nessuna risposta dall'AI")
        text = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []))
        parsed = _parse_llm_json(text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail="Errore durante la generazione dei consigli")

    suggestions = (parsed.get("movies") or [])[:7]
    out = []
    for s in suggestions:
        title = (s.get("title") or "").strip()
        year = s.get("year")
        if not title:
            continue
        try:
            params = {"query": title}
            if year:
                params["year"] = str(year)
            search = await tmdb_get("/search/movie", params)
            results = search.get("results") or []
            if results:
                m = results[0]
                out.append({**fmt_movie(m), "why": (s.get("why") or "").strip()})
            else:
                out.append({
                    "tmdb_id": None, "title": title, "poster_url": None, "backdrop_url": None,
                    "release_date": f"{year}-01-01" if year else None, "overview": None, "vote_average": 0,
                    "why": (s.get("why") or "").strip(),
                })
        except Exception:
            continue
    return {"movies": out}

# ===== Health =====
@api_router.get("/health")
async def health():
    try:
        await db.command("ping")
        mongo_ok = True
    except Exception:
        mongo_ok = False
    return {
        "status": "ok", "mongo": mongo_ok,
        "firebase": bool(firebase_admin._apps), "firebase_project_id": FIREBASE_PROJECT_ID,
    }

# ===== TMDB Routes =====
@api_router.get("/movies/popular")
@limiter.limit("60/minute")
async def movies_popular(request: Request):
    data = await tmdb_get("/movie/popular", {"page": 1})
    return [fmt_movie(m) for m in data.get("results", [])[:20]]

async def _search_actors(query: str, limit: int) -> list:
    people = await tmdb_get("/search/person", {"query": query})
    results = []
    for p in people.get("results", [])[:limit]:
        dept = p.get("known_for_department")
        if dept is not None and dept not in ("Acting", "Directing"):
            continue
        results.append({
            "id": p["id"],
            "name": p.get("name"),
            "profile_url": f"{TMDB_PROFILE_IMG}{p['profile_path']}" if p.get("profile_path") else None,
            "known_for_department": dept,
            "known_for": [k.get("title") or k.get("name") for k in p.get("known_for", [])[:3] if k.get("title") or k.get("name")],
        })
    return results

async def _search_movies(query: str) -> list:
    movies = await tmdb_get("/search/movie", {"query": query})
    seen = set()
    out = []
    for m in movies.get("results", [])[:20]:
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            out.append(fmt_movie(m))
    return out

@api_router.get("/movies/search")
@limiter.limit("30/minute")
async def movies_search(request: Request, query: str, kind: str = "auto"):
    if not query.strip():
        return {"actors": [], "movies": []}
    if len(query) > 100:
        raise HTTPException(status_code=400, detail="Query troppo lunga")
    if kind not in ("auto", "actor", "title"):
        raise HTTPException(status_code=400, detail="kind non valido")

    actors_out = []
    movies_out = []
    if kind in ("actor", "auto"):
        person_limit = 20 if kind == "actor" else 8
        actors_out = await _search_actors(query, person_limit)
    if kind in ("title", "auto"):
        movies_out = await _search_movies(query)
    return {
        "actors": actors_out[:20] if kind == "actor" else actors_out[:8],
        "movies": movies_out[:30],
    }

@api_router.get("/movies/{tmdb_id}")
@limiter.limit("60/minute")
async def movie_details(request: Request, tmdb_id: int):
    if tmdb_id <= 0 or tmdb_id > 10_000_000:
        raise HTTPException(status_code=400, detail="tmdb_id non valido")
    details = await tmdb_get(f"/movie/{tmdb_id}", {"append_to_response": "credits,videos"})
    credits = details.get("credits", {})
    cast = [
        {
            "id": c["id"], "name": c.get("name"), "character": c.get("character"),
            "profile_url": f"{TMDB_PROFILE_IMG}{c['profile_path']}" if c.get("profile_path") else None,
        }
        for c in credits.get("cast", [])[:15]
    ]
    trailer_url = pick_trailer(details.get("videos", {}))
    if not trailer_url:
        en_videos = await tmdb_get(f"/movie/{tmdb_id}/videos", {"language": "en-US"})
        trailer_url = pick_trailer(en_videos)
    movie_title = details.get("title") or details.get("original_title", "")
    providers = await fetch_watch_providers(tmdb_id, movie_title)
    return {
        **fmt_movie(details),
        "runtime": details.get("runtime"),
        "genres": [g["name"] for g in details.get("genres", [])],
        "cast": cast,
        "tagline": details.get("tagline"),
        "trailer_url": trailer_url,
        "providers": providers,
    }

async def _resolve_biography(person: dict, person_id: int) -> str:
    bio = (person.get("biography") or "").strip()
    if bio:
        return bio
    try:
        en = await tmdb_get(f"/person/{person_id}", {"language": "en-US"})
        return (en.get("biography") or "").strip()
    except Exception:
        return ""

def _build_filmography(person: dict) -> list:
    credits = person.get("movie_credits", {}) or {}
    films = [f for f in (credits.get("cast", []) or []) if f.get("release_date")]
    films.sort(key=lambda f: f.get("release_date", ""), reverse=True)
    return [{**fmt_movie(f), "character": f.get("character")} for f in films[:40]]

@api_router.get("/people/{person_id}")
@limiter.limit("60/minute")
async def person_details(request: Request, person_id: int):
    if person_id <= 0 or person_id > 10_000_000:
        raise HTTPException(status_code=400, detail="person_id non valido")
    person = await tmdb_get(f"/person/{person_id}", {"append_to_response": "movie_credits"})
    return {
        "id": person["id"],
        "name": person.get("name"),
        "biography": await _resolve_biography(person, person_id),
        "birthday": person.get("birthday"),
        "deathday": person.get("deathday"),
        "place_of_birth": person.get("place_of_birth"),
        "known_for_department": person.get("known_for_department"),
        "profile_url": f"https://image.tmdb.org/t/p/h632{person['profile_path']}" if person.get("profile_path") else None,
        "filmography": _build_filmography(person),
    }

# ===== Quiz =====
def _extract_quiz_context(details: dict) -> dict:
    credits = details.get("credits", {}) or {}
    crew = credits.get("crew", []) or []
    return {
        "title": details.get("title") or details.get("original_title", ""),
        "original_title": details.get("original_title") or details.get("title", ""),
        "year": (details.get("release_date") or "")[:4],
        "director": next((c.get("name") for c in crew if c.get("job") == "Director"), None),
        "overview": (details.get("overview") or "").strip(),
        "tagline": (details.get("tagline") or "").strip(),
        "genres": [g.get("name") for g in details.get("genres", []) if g.get("name")],
        "keywords": [k.get("name") for k in (details.get("keywords", {}) or {}).get("keywords", [])[:8] if k.get("name")],
        "cast": credits.get("cast", []) or [],
        "poster_url": f"{TMDB_IMG}{details['poster_path']}" if details.get("poster_path") else None,
    }

def _format_cast_block(cast: list) -> str:
    lines = []
    for c in cast[:8]:
        nm, ch = c.get("name"), c.get("character")
        if nm and ch:
            lines.append(f"- {nm} interpreta {ch}")
        elif nm:
            lines.append(f"- {nm}")
    return "\n".join(lines) if lines else "(non disponibile)"

def _build_quiz_prompt(ctx: dict) -> str:
    return f"""Genera un quiz e un riassunto chiave per il film:
TITOLO ITALIANO: {ctx['title']}
TITOLO ORIGINALE: {ctx['original_title']}
ANNO: {ctx['year'] or 'sconosciuto'}
REGISTA: {ctx['director'] or 'sconosciuto'}
GENERI: {', '.join(ctx['genres']) if ctx['genres'] else 'sconosciuti'}
TAGLINE: {ctx['tagline'] or '(nessuna)'}
PAROLE CHIAVE: {', '.join(ctx['keywords']) if ctx['keywords'] else '(nessuna)'}
CAST PRINCIPALE:
{_format_cast_block(ctx['cast'])}
SINOSSI UFFICIALE:
{ctx['overview']}
"""

async def _call_llm_for_quiz(tmdb_id: int, prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="LLM non configurato")
    system_msg = (
        "Sei un esperto di cinema italiano. Generi quiz a scelta multipla sulla TRAMA dei film. "
        "Rispondi SOLO con un oggetto JSON valido, senza markdown, senza testo prima o dopo."
    )
    body = {
        "systemInstruction": {"parts": [{"text": system_msg}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"},
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(GEMINI_URL, params={"key": GEMINI_API_KEY}, json=body)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="Impossibile generare il quiz")
        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise HTTPException(status_code=502, detail="Nessuna risposta dall'AI")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            raise HTTPException(status_code=502, detail="Risposta AI vuota")
        return text
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail="Impossibile generare il quiz")

def _parse_llm_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except Exception:
            raise HTTPException(status_code=502, detail="Risposta AI non valida")

def _normalize_quiz_payload(parsed: dict, ctx: dict, tmdb_id: int) -> dict:
    questions = []
    for q in (parsed.get("questions") or [])[:5]:
        questions.append({
            "id": f"plot_{len(questions) + 1}",
            "question": (q.get("question") or "").strip(),
            "options": [str(o) for o in (q.get("options") or [])[:4]],
            "correct_index": q.get("correct_index"),
            "explanation": (q.get("explanation") or "").strip(),
        })
    recap_in = parsed.get("recap") or {}
    recap = {
        "intro": ctx["title"],
        "plot": (recap_in.get("plot") or "").strip(),
        "characters": [
            {"name": (c.get("name") or "").strip(), "role": (c.get("role") or "").strip()}
            for c in (recap_in.get("characters") or [])[:6]
        ],
        "key_moments": [str(m).strip() for m in (recap_in.get("key_moments") or [])[:6]],
        "themes": [str(t).strip() for t in (recap_in.get("themes") or [])[:6]],
        "outro": ctx["tagline"],
    }
    return {
        "tmdb_id": tmdb_id,
        "title": ctx["title"],
        "poster_url": ctx["poster_url"],
        "tagline": ctx["tagline"],
        "questions": questions,
        "recap": recap,
        "ai_generated": True,
    }

def _empty_quiz_payload(ctx: dict, tmdb_id: int) -> dict:
    return {
        "tmdb_id": tmdb_id, "title": ctx["title"], "poster_url": ctx["poster_url"],
        "tagline": ctx["tagline"], "questions": [],
        "recap": {
            "intro": f"Trama di «{ctx['title']}» non disponibile in italiano su TMDB.",
            "plot": "", "characters": [], "key_moments": [], "themes": [],
            "outro": ctx["tagline"] or "",
        },
        "ai_generated": False,
    }

async def _cache_quiz(tmdb_id: int, payload: dict) -> None:
    if not payload.get("questions") or not payload["recap"].get("plot"):
        return
    try:
        await db.movie_quizzes.update_one(
            {"tmdb_id": tmdb_id, "lang": "it"},
            {"$set": {
                "tmdb_id": tmdb_id, "lang": "it", "version": 2,
                "payload": payload, "generated_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )
    except Exception:
        pass

@api_router.get("/quiz/{tmdb_id}")
@limiter.limit("20/minute")
async def movie_quiz(request: Request, tmdb_id: int):
    if tmdb_id <= 0 or tmdb_id > 10_000_000:
        raise HTTPException(status_code=400, detail="tmdb_id non valido")
    cached = await db.movie_quizzes.find_one(
        {"tmdb_id": tmdb_id, "lang": "it", "version": 2}, {"_id": 0}
    )
    if cached:
        return cached.get("payload", cached)
    try:
        details = await tmdb_get(f"/movie/{tmdb_id}", {"append_to_response": "credits,keywords"})
    except HTTPException:
        raise HTTPException(status_code=404, detail="Film non trovato")

    ctx = _extract_quiz_context(details)
    
    # IL FIX E' QUI: Se la trama in italiano non esiste, la scarica in inglese per far lavorare Gemini
    if not ctx["overview"] or len(ctx["overview"]) < 30:
        try:
            en_details = await tmdb_get(f"/movie/{tmdb_id}", {"language": "en-US", "append_to_response": "credits,keywords"})
            ctx["overview"] = _extract_quiz_context(en_details).get("overview", "")
        except Exception:
            pass
            
    if not ctx["overview"] or len(ctx["overview"]) < 30:
        return _empty_quiz_payload(ctx, tmdb_id)

    raw = await _call_llm_for_quiz(tmdb_id, _build_quiz_prompt(ctx))
    parsed = _parse_llm_json(raw)
    payload = _normalize_quiz_payload(parsed, ctx, tmdb_id)
    await _cache_quiz(tmdb_id, payload)
    return payload

@api_router.get("/")
async def root():
    return {"message": "CineDiario API", "status": "ok"}

app.include_router(api_router)

if ALLOWED_HOSTS and ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

cors_origins = ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"]
cors_creds = bool(ALLOWED_ORIGINS) 
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_creds,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BodySizeLimitMiddleware)

@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    if IS_PROD:
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

@app.on_event("startup")
async def startup_db():
    await db.users.create_index("user_id", unique=True)
    await db.users.create_index("email", sparse=True)
    await db.users.create_index("friend_code", unique=True, sparse=True)
    await db.user_movies.create_index([("user_id", 1), ("tmdb_id", 1)], unique=True)
    await db.user_movies.create_index([("user_id", 1), ("status", 1), ("updated_at", -1)])
    await db.friendships.create_index([("user_lo", 1), ("user_hi", 1)], unique=True)
    await db.friendships.create_index("request_id", unique=True, sparse=True)
    await db.shares.create_index("share_id", unique=True)
    await db.shares.create_index([("to_user_id", 1), ("read", 1), ("created_at", -1)])
    logger.info("CineDiario backend ready (Firebase Auth + MongoDB Atlas + Security hardened)")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
# ... [mantiene tutti i tuoi import esistenti] ...

# ===== USER MOVIES (Snapshot corretto) =====
async def _movie_snapshot(tmdb_id: int) -> dict:
    try:
        details = await tmdb_get(f"/movie/{tmdb_id}")
        return {
            "tmdb_id": details["id"],
            "title": details.get("title") or details.get("original_title", ""),
            "poster_url": f"{TMDB_IMG}{details['poster_path']}" if details.get("poster_path") else None,
            "backdrop_url": f"{TMDB_IMG}{details['backdrop_path']}" if details.get("backdrop_path") else None,
            "release_date": details.get("release_date"),
            "overview": details.get("overview"),
            "vote_average": details.get("vote_average", 0),
            "genres": [g.get("name") for g in details.get("genres", []) if g.get("name")],
            "runtime": details.get("runtime"),
        }
    except Exception:
        return {"tmdb_id": tmdb_id, "title": "", "poster_url": None}

def _serialize_user_movie(doc: dict) -> dict:
    movie_obj = doc.get("movie") or {}
    return {
        "tmdb_id": doc["tmdb_id"],
        "status": doc.get("status"),
        "rating": doc.get("rating"),
        "overall": doc.get("overall"),
        "title": movie_obj.get("title", ""),
        "poster_url": movie_obj.get("poster_url"),
        "backdrop_url": movie_obj.get("backdrop_url"),
        "notes": doc.get("notes"),
        "movie": movie_obj,
        "created_at": (doc.get("created_at") or datetime.now(timezone.utc)).isoformat(),
        "updated_at": (doc.get("updated_at") or datetime.now(timezone.utc)).isoformat(),
    }

# ===== Statistiche (Corretto per Profile.tsx) =====
@api_router.get("/user/stats")
@limiter.limit("30/minute")
async def user_stats(request: Request, user: dict = Depends(get_current_user)):
    # Statistiche generali
    pipeline = [
        {"$match": {"user_id": user["user_id"]}},
        {"$group": {
            "_id": "$status",
            "count": {"$sum": 1},
            "avg_rating": {"$avg": "$overall"},
        }},
    ]
    # Distribuzione voti (1-10)
    dist_pipeline = [
        {"$match": {"user_id": user["user_id"], "status": "watched", "overall": {"$exists": True}}},
        {"$group": {"_id": "$overall", "count": {"$sum": 1}}},
    ]
    
    by_status = {}
    total = 0
    sum_minutes = 0
    async for row in db.user_movies.aggregate(pipeline):
        by_status[row["_id"] or "unknown"] = {
            "count": row["count"],
            "avg_rating": round(row["avg_rating"], 2) if row["avg_rating"] else None,
        }
        total += row["count"]
        
    rating_dist = {}
    async for row in db.user_movies.aggregate(dist_pipeline):
        rating_dist[str(int(row["_id"] or 0))] = row["count"]

    runtime_cursor = db.user_movies.find(
        {"user_id": user["user_id"], "status": "watched"},
        {"_id": 0, "movie.runtime": 1},
    )
    async for d in runtime_cursor:
        rt = ((d.get("movie") or {}).get("runtime")) or 0
        sum_minutes += rt
        
    return {
        "total_watched": by_status.get("watched", {}).get("count", 0),
        "total_watchlist": by_status.get("watchlist", {}).get("count", 0),
        "average_rating": by_status.get("watched", {}).get("avg_rating"),
        "total": total,
        "rating_distribution": rating_dist,
        "watched_minutes": sum_minutes,
        "watched_hours": round(sum_minutes / 60, 1),
    }

# ===== AI MOVIE FINDER (Ripristinata rotta vecchia + nuova) =====
@api_router.post("/ai/movie-finder") # Aggiunta rotta vecchia per compatibilità
@api_router.post("/discover/ai-recommend") # Mantenuta rotta nuova
@limiter.limit("15/minute")
async def ai_movie_finder(request: Request, req: MovieFinderReq, user: dict = Depends(get_current_user)):
    # ... [Inserisci qui il codice della funzione ai_movie_finder identico a quello che avevi] ...
    # (Il resto del file rimane identico a quello che ti ho dato prima)
