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
import asyncio
from datetime import datetime, timezone
import httpx
import secrets
import json
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from firebase_admin import messaging

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

# ============================================================
# HELPER PUSH EXPO con data per il deep-link
# ============================================================
async def send_expo_push(http_client, token: str, title: str, body: str, data: dict | None = None):
    if not token or not token.startswith("ExponentPushToken"):
        return
    payload = {
        "to": token,
        "title": title,
        "body": body,
        "sound": "default",
        "priority": "high",
        "channelId": "default",
        "data": data or {},
    }
    try:
        await http_client.post(
            "https://exp.host/--/api/v2/push/send",
            json=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=10.0,
        )
    except Exception as e:
        logger.error(f"Errore invio push: {e}")

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
    email = user.get("email", "").strip()
    if not email:
        is_guest = True
    else:
        is_guest = user.get("auth_provider_id") == "anonymous" or user.get("auth_provider") == "guest"
        
    return {
        "user_id": user["user_id"],
        "email": email,
        "name": user.get("name", ""),
        "picture": user.get("picture"),
        "is_guest": is_guest,
        "friend_code": user.get("friend_code") if not is_guest else None,
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
            sign_in_provider = "anonymous"
            if ur.provider_data:
                sign_in_provider = ur.provider_data[0].provider_id

            return {
                "uid": ur.uid,
                "email": (ur.email or "").lower(),
                "email_verified": bool(ur.email_verified),
                "name": ur.display_name or "",
                "picture": ur.photo_url,
                "aud": FIREBASE_PROJECT_ID,
                "iss": expected_iss,
                "firebase": {"sign_in_provider": sign_in_provider},
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
            "auth_provider_id": profile.get("provider_id"),
            "created_at": now,
            "last_login_at": now,
        }
        try:
            await ensure_friend_code(user)
            await db.users.insert_one(user)
            security_logger.info(f"AUTH_NEW_USER uid={uid} provider={profile.get('provider_id')}")
        except Exception:
            user = await db.users.find_one({"user_id": uid}, {"_id": 0}) or user
    else:
        updates = {"last_login_at": now}
        if profile.get("provider_id") and user.get("auth_provider_id") != profile["provider_id"]:
            updates["auth_provider_id"] = profile["provider_id"]
        for key in ("name", "picture", "email", "email_verified"):
            if profile.get(key) and user.get(key) != profile[key]:
                updates[key] = profile[key]
        
        await ensure_friend_code(user)
        updates["friend_code"] = user["friend_code"]
        await db.users.update_one({"user_id": uid}, {"$set": updates})
        user.update(updates)
        
    return user

async def get_current_user(request: Request, authorization: Optional[str] = Header(None)) -> dict:
    client_ip = get_remote_address(request)
    token = _extract_bearer_token(authorization, client_ip)
    decoded = _verify_firebase_token(token, client_ip)
    profile = _build_user_profile(decoded)
    return await _upsert_user(profile)

async def tmdb_get(path: str, params: dict = None, cache_hours: int = 48) -> dict:
    p = {"language": TMDB_LANG}
    if params:
        p.update(params)
        
    # 1. Generiamo una "chiave" unica basata su cosa l'utente ha cercato
    # Es: "/search/movie?language=it-IT&query=Inception"
    query_string = "&".join(f"{k}={v}" for k, v in sorted(p.items()))
    cache_key = f"{path}?{query_string}"
    
    now = datetime.now(timezone.utc)
    
    # 2. Controlliamo se abbiamo GIA' questa risposta salvata nel database
    try:
        cached = await db.tmdb_cache.find_one({"_id": cache_key})
        if cached:
            # Controlliamo che la ricerca non sia più vecchia di 48 ore
            delta = (now - cached["updated_at"]).total_seconds()
            if delta < (cache_hours * 3600):
                return cached["data"]
    except Exception as e:
        logger.error(f"Errore lettura cache TMDB: {e}")

    # 3. Se NON c'è, o è scaduta, chiediamo i dati veri a TMDB
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get(f"{TMDB_BASE}{path}", headers=TMDB_HEADERS, params=p)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"TMDB error: {r.status_code}")
        
        data = r.json()
        
        # 4. Salviamo i dati freschi nel Database per i prossimi utenti!
        try:
            await db.tmdb_cache.update_one(
                {"_id": cache_key},
                {"$set": {"data": data, "updated_at": now}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Errore scrittura cache TMDB: {e}")
            
        return data

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

@api_router.get("/auth/me")
@limiter.limit("30/minute")
async def auth_me(request: Request, user: dict = Depends(get_current_user)):
    return {"user": serialize_user(user)}

class PushTokenReq(BaseModel):
    token: str

@api_router.post("/user/push-token")
@limiter.limit("5/minute")
async def save_push_token(request: Request, req: PushTokenReq, user: dict = Depends(get_current_user)):
    await db.users.update_one(
        {"user_id": user["user_id"]}, 
        {"$set": {"push_token": req.token}}
    )
    return {"ok": True}

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
            "production_countries": [c.get("name") for c in details.get("production_countries", []) if c.get("name")],
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
        "rating_directing": doc.get("rating_directing"),
        "rating_acting": doc.get("rating_acting"),
        "rating_screenplay": doc.get("rating_screenplay"),
        "rating_soundtrack": doc.get("rating_soundtrack"),
        "rating_cinematography": doc.get("rating_cinematography"),
        "movie": movie_obj,
        "created_at": (doc.get("created_at") or datetime.now(timezone.utc)).isoformat(),
        "updated_at": (doc.get("updated_at") or datetime.now(timezone.utc)).isoformat(),
        "genres": movie_obj.get("genres", []),
        "production_countries": movie_obj.get("production_countries", []),
        "release_date": movie_obj.get("release_date"),
    }

@api_router.post("/user/movies/fix-metadata")
@limiter.limit("5/minute")
async def fix_old_movies_metadata(request: Request, user: dict = Depends(get_current_user)):
    cursor = db.user_movies.find({
        "user_id": user["user_id"],
        "$or": [
            {"movie.production_countries": {"$exists": False}},
            {"movie.production_countries": None}
        ]
    })
    updated_count = 0
    async for doc in cursor:
        tmdb_id = doc["tmdb_id"]
        new_snapshot = await _movie_snapshot(tmdb_id)
        await db.user_movies.update_one(
            {"user_id": user["user_id"], "tmdb_id": tmdb_id},
            {"$set": {"movie": new_snapshot}}
        )
        updated_count += 1
    return {"ok": True, "updated_count": updated_count}

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
    
    # --- AGGIUNTA FEED PROTETTA: Registra il film visto ---
    try:
        if req.status == "watched":
            m_title = snapshot.get("title") if snapshot else "un film"
            await log_activity(user["user_id"], "watch", str(req.tmdb_id), str(m_title))
    except Exception as e:
        print(f"Errore feed upsert: {e}")

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
    if not existing.get("movie") or not existing.get("movie", {}).get("poster_url"):
        updates["movie"] = await _movie_snapshot(tmdb_id)
        
    await db.user_movies.update_one({"user_id": user["user_id"], "tmdb_id": tmdb_id}, {"$set": updates})
    doc = await db.user_movies.find_one({"user_id": user["user_id"], "tmdb_id": tmdb_id}, {"_id": 0})
    
  # --- AGGIUNTA FEED PROTETTA: Voto / Aggiornamento ---
    try:
        if updates.get("status") == "watched" or existing.get("status") == "watched":
            # Estraiamo il titolo in modo ipersicuro, evitando crash se i dati mancano
            m_dict = doc.get("movie") or {}
            m_title = m_dict.get("title") or "un film"
            await log_activity(user["user_id"], "watch", str(tmdb_id), str(m_title))
    except Exception as e:
        print(f"Errore feed update: {e}")

    return _serialize_user_movie(doc)

@api_router.delete("/user/movies/{tmdb_id}")
@limiter.limit("30/minute")
async def delete_user_movie(request: Request, tmdb_id: int, user: dict = Depends(get_current_user)):
    result = await db.user_movies.delete_one({"user_id": user["user_id"], "tmdb_id": tmdb_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Film non presente nel diario")
    return {"ok": True}

# ==========================================
# GESTIONE ATTORI SEGUITI E STATISTICHE
# ==========================================

class FollowPersonReq(BaseModel):
    name: str
    profile_url: Optional[str] = None

@api_router.post("/user/people/{person_id}/follow")
@limiter.limit("20/minute")
async def follow_person(request: Request, person_id: int, req: FollowPersonReq, user: dict = Depends(get_current_user)):
    doc = {
        "user_id": user["user_id"],
        "person_id": person_id,
        "name": req.name,
        "profile_url": req.profile_url,
        "created_at": datetime.now(timezone.utc)
    }
    await db.user_people.update_one(
        {"user_id": user["user_id"], "person_id": person_id},
        {"$set": doc},
        upsert=True
    )
    
    # --- AGGIUNTA FEED PROTETTA: Segui un attore ---
    try:
        p_name = getattr(req, "name", "un attore")
        await log_activity(user["user_id"], "follow", str(person_id), str(p_name))
    except Exception:
        pass
        
    return {"ok": True, "status": "followed"}

@api_router.delete("/user/people/{person_id}/unfollow")
@limiter.limit("20/minute")
async def unfollow_person(request: Request, person_id: int, user: dict = Depends(get_current_user)):
    await db.user_people.delete_one({"user_id": user["user_id"], "person_id": person_id})
    return {"ok": True, "status": "unfollowed"}

@api_router.get("/user/people/{person_id}/status")
@limiter.limit("30/minute")
async def check_person_status(request: Request, person_id: int, user: dict = Depends(get_current_user)):
    doc = await db.user_people.find_one({"user_id": user["user_id"], "person_id": person_id})
    return {"is_following": bool(doc)}

# ==========================================
# PROFILI PUBBLICI E FILM IN EVIDENZA
# ==========================================

class HighlightedMovie(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None

class HighlightedMoviesReq(BaseModel):
    movies: List[HighlightedMovie]

@api_router.post("/user/highlighted-movies")
@limiter.limit("20/minute")
async def update_highlighted_movies(request: Request, req: HighlightedMoviesReq, user: dict = Depends(get_current_user)):
    # Limitiamo a un massimo di 4 film in evidenza per non intasare la UI
    movies_to_save = [m.model_dump() for m in req.movies[:4]]
    
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": {"highlighted_movies": movies_to_save}}
    )
    
    # --- AGGIUNTA FEED PROTETTA: Film in vetrina ---
    try:
        if hasattr(req, "movies") and req.movies:
            for m in req.movies:
                m_id = getattr(m, "tmdb_id", "0")
                m_title = getattr(m, "title", "un film")
                await log_activity(user["user_id"], "showcase", str(m_id), str(m_title))
    except Exception:
        pass
        
    return {"ok": True, "highlighted_movies": movies_to_save}

@api_router.get("/friends/{friend_id}/profile")
@limiter.limit("30/minute")
async def get_friend_profile(request: Request, friend_id: str, user: dict = Depends(get_current_user)):
    # 1. Controlla che siano effettivamente amici (privacy)
    if not await _are_friends(user["user_id"], friend_id):
        raise HTTPException(status_code=403, detail="Non sei amico di questo utente")

    # 2. Recupera i dati base e i film in evidenza dell'amico
    target_user = await db.users.find_one(
        {"user_id": friend_id}, 
        {"_id": 0, "name": 1, "picture": 1, "highlighted_movies": 1, "created_at": 1}
    )
    if not target_user:
        raise HTTPException(status_code=404, detail="Utente non trovato")

    # 3. Recupera gli attori seguiti dall'amico
    followed_cursor = db.user_people.find({"user_id": friend_id}, {"_id": 0}).sort("created_at", -1)
    followed_actors = [doc async for doc in followed_cursor]

    # 4. Calcola quanti film ha visto in totale (per fargli fare un po' di scena)
    total_watched = await db.user_movies.count_documents({"user_id": friend_id, "status": "watched"})

    return {
        "user_id": friend_id,
        "name": target_user.get("name", "Utente"),
        "picture": target_user.get("picture"),
        "member_since": target_user.get("created_at").isoformat() if target_user.get("created_at") else None,
        "total_watched": total_watched,
        "highlighted_movies": target_user.get("highlighted_movies", []),
        "followed_actors": followed_actors
    }

# ==========================================
# AVATAR DA TMDB
# ==========================================

class AvatarReq(BaseModel):
    picture_url: str

@api_router.post("/user/avatar")
@limiter.limit("10/minute")
async def update_avatar(request: Request, req: AvatarReq, user: dict = Depends(get_current_user)):
    # Aggiorna il campo 'picture' nel documento dell'utente
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": {"picture": req.picture_url}}
    )
    return {"ok": True, "picture": req.picture_url}

@api_router.get("/search-avatar")
@limiter.limit("20/minute")
async def search_avatar(request: Request, q: str, user: dict = Depends(get_current_user)):
    # 1. Cerchiamo su TMDB come sempre
    try:
        data = await tmdb_get("/search/multi", {"query": q, "page": "1"})
    except Exception as e:
        data = {"results": []}
        
    results = []
    for item in data.get("results", []):
        if item.get("media_type") in ["movie", "person"]:
            path = item.get("profile_path") if item.get("media_type") == "person" else item.get("poster_path")
            if path:
                results.append({
                    "id": item.get("id"),
                    "name": item.get("title") if item.get("media_type") == "movie" else item.get("name"),
                    "type": "Film" if item.get("media_type") == "movie" else "Persona",
                    "image_url": f"https://image.tmdb.org/t/p/w200{path}"
                })
                
    suggestion = None
    
    # 2. IL TRUCCO WIKIPEDIA (Con il "Documento di Identità")
    if not results:
        import httpx
        import urllib.parse
        try:
            safe_q = urllib.parse.quote(q)
            # Usiamo Wikipedia Italia per avere correzioni migliori sui nomi
            wiki_url = f"https://it.wikipedia.org/w/api.php?action=query&list=search&srsearch={safe_q}&srinfo=suggestion&format=json"
            
            # IL DOCUMENTO D'IDENTITÀ: Senza questo, Wikipedia ci blocca credendoci degli hacker!
            headers = {
                "User-Agent": "CineDiarioApp/1.0 (info@cinediario.app)"
            }
            
            # Mandiamo la richiesta a Wikipedia presentando il documento
            async with httpx.AsyncClient(timeout=3.0, headers=headers) as client:
                w_resp = await client.get(wiki_url)
                
                # Se Wikipedia ci accetta, peschiamo il suggerimento
                if w_resp.status_code == 200:
                    w_data = w_resp.json()
                    suggestion = w_data.get("query", {}).get("searchinfo", {}).get("suggestion")
                    
                    # Estetica: prima lettera maiuscola
                    if suggestion:
                        suggestion = suggestion.title()
                else:
                    print(f"Wikipedia ha rifiutato la connessione con codice: {w_resp.status_code}")
                    
        except Exception as e:
            print(f"Errore di connessione a Wikipedia: {e}")
            pass 
            
    return {"results": results[:12], "suggestion": suggestion}

# ==========================================
# FEED SOCIALE - Attività
# ==========================================

async def log_activity(user_id: str, action_type: str, target_id: str, target_name: str, meta: dict = {}):
    try:
        user_doc = await db.users.find_one({"user_id": user_id}, {"name": 1})
        user_name = user_doc.get("name") if user_doc else "Utente"
        activity = {
            "user_id": user_id,
            "user_name": user_name,
            "action_type": action_type, 
            "target_id": str(target_id),
            "target_name": target_name,
            "meta": meta,
            "created_at": datetime.now(timezone.utc) # ORA È CONSAPEVOLE DEL FUSO ORARIO
        }
        await db.activity_log.insert_one(activity)
    except Exception as e:
        print(f"Errore durante il salvataggio del log: {e}")

@api_router.get("/feed")
@limiter.limit("30/minute")
async def get_feed(request: Request, user: dict = Depends(get_current_user)):
    me = user["user_id"]
    
    cursor = db.friendships.find(
        {"$or": [{"user_lo": me}, {"user_hi": me}], "status": "accepted"},
        {"_id": 0, "user_lo": 1, "user_hi": 1}
    )
    
    friends = [me] 
    async for fr in cursor:
        other_id = fr["user_hi"] if fr["user_lo"] == me else fr["user_lo"]
        friends.append(other_id)
    
    activities = await db.activity_log.find({"user_id": {"$in": friends}}) \
        .sort("created_at", -1) \
        .limit(30) \
        .to_list(length=30)
        
    for act in activities:
        act["_id"] = str(act["_id"])
        if "created_at" in act and hasattr(act["created_at"], "isoformat"):
            dt_str = act["created_at"].isoformat()
            # Se la data non finisce con Z o non ha il fuso, la forziamo in UTC
            if not dt_str.endswith('Z') and '+' not in dt_str:
                dt_str += 'Z'
            act["created_at"] = dt_str
            
    return {"activities": activities}

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
    dist_pipeline = [
        {"$match": {"user_id": user["user_id"], "status": "watched", "overall": {"$exists": True, "$ne": None}}},
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
        val = row.get("_id")
        if val is not None:
            try:
                bucket = str(int(float(val)))
                rating_dist[bucket] = rating_dist.get(bucket, 0) + row.get("count", 0)
            except (ValueError, TypeError):
                pass

    runtime_cursor = db.user_movies.find(
        {"user_id": user["user_id"], "status": "watched"},
        {"_id": 0, "movie.runtime": 1},
    )
    async for d in runtime_cursor:
        rt = ((d.get("movie") or {}).get("runtime")) or 0
        sum_minutes += rt
        
    # RECUPERA GLI ATTORI SEGUITI E LI METTE NELLE STATISTICHE
    followed_cursor = db.user_people.find({"user_id": user["user_id"]}, {"_id": 0}).sort("created_at", -1)
    followed_actors = [doc async for doc in followed_cursor]
        
    return {
        "total_watched": by_status.get("watched", {}).get("count", 0),
        "total_watchlist": by_status.get("watchlist", {}).get("count", 0),
        "average_rating": by_status.get("watched", {}).get("avg_rating"),
        "total": total,
        "by_status": by_status,
        "rating_distribution": rating_dist,
        "watched_minutes": sum_minutes,
        "watched_hours": round(sum_minutes / 60, 1),
        "followed_actors": followed_actors,
        "highlighted_movies": user.get("highlighted_movies", []),
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
        return {
            "recommendations": recs[:20], 
            "message": "Valuta almeno un film per ricevere suggerimenti personalizzati"
        }

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
    return {"friends": out}

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
            "request_id": fr.get("request_id"),
            "created_at": (fr.get("created_at") or datetime.now(timezone.utc)).isoformat(),
        }
        if fr.get("requested_by") == me:
            entry["to_user"] = other_profile
            outgoing.append(entry)
        else:
            entry["from_user"] = other_profile
            incoming.append(entry)
    return {"incoming": incoming, "outgoing": outgoing}

class FriendRequestReq(BaseModel):
    friend_code: str

@api_router.post("/friends/request")
@limiter.limit("10/minute")
async def send_friend_request(request: Request, req: FriendRequestReq, user: dict = Depends(get_current_user)):
    code = req.friend_code.strip().upper()
    target = await db.users.find_one({"friend_code": code})
    if not target:
        raise HTTPException(status_code=404, detail="Codice amico non trovato")
    
    target_id = target["user_id"]
    if target_id == user["user_id"]:
        raise HTTPException(status_code=400, detail="Non puoi aggiungere te stesso")

    user_lo, user_hi = (user["user_id"], target_id) if user["user_id"] < target_id else (target_id, user["user_id"])
    existing = await db.friendships.find_one({"user_lo": user_lo, "user_hi": user_hi})
    
    if existing:
        raise HTTPException(status_code=400, detail="Relazione già esistente")

    request_id = f"fr_{uuid.uuid4().hex[:12]}"
    await db.friendships.insert_one({
        "user_lo": user_lo,
        "user_hi": user_hi,
        "request_id": request_id,
        "requested_by": user["user_id"],
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
    })
    
    profile = await _friend_profile(target_id)
    safe_profile = profile if profile else {"name": "Utente", "user_id": target_id}
    return {"ok": True, "request_id": request_id, "to": safe_profile}

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
    
    other_id = _other_user(fr, me)
    profile = await _friend_profile(other_id)
    safe_profile = profile if profile else {"name": "Utente", "user_id": other_id}
    return {"ok": True, "friend": safe_profile}

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

class ShareReq(BaseModel):
    to_user_ids: List[str] 
    tmdb_id: int
    share_type: str = "recommendation" 
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
        movie_data = s.get("movie_snapshot") or s.get("movie") or {}
        out.append({
            "share_id": s["share_id"],
            "from_user": await _friend_profile(s["from_user_id"]),
            "tmdb_id": s["tmdb_id"],
            "share_type": s.get("share_type", "recommendation"),
            "movie_snapshot": movie_data,
            "review_snapshot": s.get("review_snapshot"),
            "message": s.get("message"),
            "read": bool(s.get("read", False)),
            "created_at": (s.get("created_at") or datetime.now(timezone.utc)).isoformat(),
        })
    return {"items": out}

@api_router.post("/shares")
@limiter.limit("20/minute")
async def create_share(request: Request, req: ShareReq, user: dict = Depends(get_current_user)):
    if user.get("auth_provider") == "guest":
        raise HTTPException(status_code=403, detail="Crea un account per condividere")
    if not req.to_user_ids:
        raise HTTPException(status_code=400, detail="Seleziona almeno un amico")

    me = user["user_id"]
    for tid in req.to_user_ids:
        if not await _are_friends(me, tid):
            raise HTTPException(status_code=403, detail="Puoi condividere solo con i tuoi amici")

    movie = await tmdb_get(f"/movie/{req.tmdb_id}")
    movie_snap = {
        "tmdb_id": movie.get("id"),
        "title": movie.get("title") or movie.get("original_title", ""),
        "poster_url": f"{TMDB_IMG}{movie['poster_path']}" if movie.get("poster_path") else None,
        "release_date": movie.get("release_date"),
        "vote_average": movie.get("vote_average"),
    }
    
    review_snap = None
    if req.share_type == "review":
        item = await db.user_movies.find_one({"user_id": me, "tmdb_id": req.tmdb_id, "status": "watched"})
        if item:
            review_snap = {
                "rating": item.get("rating"),
                "notes": item.get("notes"),
                "rating_directing": item.get("rating_directing"),
                "rating_acting": item.get("rating_acting"),
                "rating_screenplay": item.get("rating_screenplay"),
                "rating_soundtrack": item.get("rating_soundtrack"),
                "rating_cinematography": item.get("rating_cinematography"),
            }

    docs = [{
        "share_id": uuid.uuid4().hex,
        "from_user_id": me,
        "to_user_id": tid,
        "tmdb_id": req.tmdb_id,
        "share_type": req.share_type,
        "message": (req.message or "").strip()[:500] or None,
        "movie_snapshot": movie_snap,
        "review_snapshot": review_snap,
        "read": False,
        "created_at": datetime.now(timezone.utc),
    } for tid in req.to_user_ids]
    
    await db.shares.insert_many(docs)
    
    async with httpx.AsyncClient() as client:
        for tid in req.to_user_ids:
            target_user = await db.users.find_one({"user_id": tid})
            if target_user:
                token_dispositivo = target_user.get("push_token")
                if token_dispositivo:
                    await send_expo_push(
                        client,
                        token=token_dispositivo,
                        title="🎬 Nuovo film condiviso!",
                        body=f"{user.get('name', 'Qualcuno')} ti ha inviato «{movie_snap['title']}»",
                        data={
                            "type": "movie_share",
                            "tmdb_id": req.tmdb_id,
                            "screen": "movie",
                        },
                    )
    return {"ok": True, "sent": len(docs)}

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

class MovieFinderReq(BaseModel):
    answers: dict
    free_text: Optional[str] = None

@api_router.post("/discover/ai-recommend")
@api_router.post("/ai/movie-finder")
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

ATTENZIONE - REGOLA FERREA SULLE PIATTAFORME:
Se l'utente indica "Amazon", "Prime", "Prime Video" o simili tra le piattaforme, devi consigliare ESCLUSIVAMENTE film che sono inclusi GRATUITAMENTE nell'abbonamento Prime Video Italia. È severamente vietato suggerire film che su Amazon sono disponibili solo a noleggio o per l'acquisto.
Applica questa regola a tutte le piattaforme: suggerisci solo film inclusi nell'abbonamento base (streaming flatrate).

Restituisci SOLO un oggetto JSON con questo formato esatto:
{{"movies": [{{"title": "Titolo italiano del film", "year": 2010, "why": "Breve motivo (max 25 parole) in italiano"}}, ...]}}
Esattamente 7 film, in italiano, vari per stile ma coerenti con le risposte.
Non aggiungere testo prima o dopo il JSON, non usare markdown."""

    body = {
        "systemInstruction": {"parts": [{"text": "Sei un esperto cinefilo italiano. Consigli film esistenti reali, mai inventati. Rispondi solo JSON."}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.8, "responseMimeType": "application/json"},
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }
    
    try:
        # Alzato il timeout a 45 secondi
        async with httpx.AsyncClient(timeout=45.0) as c:
            r = await c.post(GEMINI_URL, params={"key": GEMINI_API_KEY}, json=body)
        if r.status_code != 200:
            logger.error(f"Errore API AI: {r.text}")
            raise HTTPException(status_code=502, detail="Impossibile generare consigli (Errore AI)")
        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            logger.error(f"Risposta AI bloccata: {data}")
            raise HTTPException(status_code=502, detail="Nessuna risposta dall'AI (Filtri di sicurezza)")
        text = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []))
        parsed = _parse_llm_json(text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Errore connessione AI: {e}")
        raise HTTPException(status_code=502, detail="Errore durante la generazione (Riprova)")

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
                # Se il film non esiste, lo salta direttamente senza creare "fantasmi"
                continue
        except Exception:
            continue
    return {"movies": out}

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

ISTRUZIONI:
1. Genera ESATTAMENTE 5 domande a scelta multipla (4 opzioni ciascuna) sulla TRAMA, sui PERSONAGGI e su quello che ACCADE nel film.
   - NON fare domande su: anno di uscita, durata, regista, paese di produzione, premi, incassi.
   - SI a domande tipo: motivazioni dei personaggi, conflitti, scelte morali, eventi chiave, finale, relazioni tra personaggi, ambientazione narrativa, oggetti/luoghi simbolici.
   - Le domande devono essere CHIARE e RISOLVIBILI da chi ha visto il film (anche tempo fa).
   - I distrattori devono essere PLAUSIBILI ma chiaramente SBAGLIATI per chi ricorda il film.
   - Lingua: italiano, tono naturale.

2. Genera un RIASSUNTO CHIAVE strutturato per aiutare lo spettatore a ricordare il film:
   - "plot": un riassunto narrativo di 4-6 frasi che ripercorre inizio, sviluppo, climax e finale.
   - "characters": 3-5 personaggi principali con 1 frase ciascuno (nome personaggio + ruolo nella storia).
   - "key_moments": 3-5 scene/momenti chiave da ricordare (frasi brevi e vivide).
   - "themes": 2-4 temi principali del film.

FORMATO DI RISPOSTA (JSON ESATTO):
{{
  "questions": [
    {{
      "question": "...",
      "options": ["a", "b", "c", "d"],
      "correct_index": 0,
      "explanation": "breve spiegazione del perché è la risposta giusta (1 frase)"
    }}
  ],
  "recap": {{
    "plot": "...",
    "characters": [
      {{"name": "Nome Personaggio", "role": "ruolo nella storia in una frase"}}
    ],
    "key_moments": ["momento 1", "momento 2"],
    "themes": ["tema 1", "tema 2"]
  }}
}}

Restituisci SOLO l'oggetto JSON valido.
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

class PublicReviewReq(BaseModel):
    text: Optional[str] = None
    rating: Optional[float] = None
    is_anonymous: bool = False

class ReplyReq(BaseModel):
    text: str

@api_router.post("/movies/{tmdb_id}/public-reviews")
@limiter.limit("10/minute")
async def add_public_review(request: Request, tmdb_id: int, req: PublicReviewReq, user: dict = Depends(get_current_user)):
    if user.get("auth_provider") == "guest":
        raise HTTPException(status_code=403, detail="Crea un account per lasciare recensioni pubbliche")

    text_stripped = req.text.strip() if req.text else ""
    if not text_stripped and req.rating is None:
        raise HTTPException(status_code=400, detail="Inserisci un voto o un commento")

    now = datetime.now(timezone.utc)

    # Prepara i dati da aggiornare
    updates = {
        "user_name": user.get("name", "Utente"),
        "user_picture": user.get("picture"),
        "text": text_stripped[:1000] if text_stripped else None,
        "rating": req.rating,
        "is_anonymous": req.is_anonymous,
        "updated_at": now
    }

    # Salviamo la recensione. $setOnInsert protegge i "likes" se la recensione esiste già!
    result = await db.public_reviews.update_one(
        {"tmdb_id": tmdb_id, "user_id": user["user_id"]},
        {
            "$set": updates,
            "$setOnInsert": {
                "review_id": uuid.uuid4().hex,
                "created_at": now,
                "likes": [] 
            }
        },
        upsert=True
    )

    # --- AGGIUNTA FEED: Logga il voto o commento SOLO se è nuovo (non una modifica) e non anonimo ---
    if not req.is_anonymous and result.matched_count == 0:
        try:
            snap = await _movie_snapshot(tmdb_id)
            m_title = snap.get("title", "un film")
            await log_activity(user["user_id"], "review", str(tmdb_id), str(m_title))
        except Exception as e:
            print(f"Errore feed review: {e}")

    return {"ok": True}

@api_router.post("/movies/{tmdb_id}/public-reviews/{review_id}/reply")
@limiter.limit("15/minute")
async def add_review_reply(request: Request, tmdb_id: int, review_id: str, req: ReplyReq, user: dict = Depends(get_current_user)):
    if user.get("auth_provider") == "guest":
        raise HTTPException(status_code=403, detail="Crea un account per rispondere")

    text_stripped = req.text.strip()
    if not text_stripped:
        raise HTTPException(status_code=400, detail="Il testo non può essere vuoto")

    reply = {
        "reply_id": uuid.uuid4().hex,
        "user_id": user["user_id"],
        "user_name": user.get("name", "Utente"),
        "user_picture": user.get("picture"),
        "text": text_stripped[:500],
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    result = await db.public_reviews.update_one(
        {"tmdb_id": tmdb_id, "review_id": review_id},
        {"$push": {"replies": reply}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Recensione non trovata")

    return {"ok": True, "reply": reply}

# UNICA VERSIONE DI DELETE_REVIEW
@api_router.delete("/movies/{tmdb_id}/public-reviews/{review_id}")
@limiter.limit("20/minute")
async def delete_review(request: Request, tmdb_id: int, review_id: str, user: dict = Depends(get_current_user)):
    result = await db.public_reviews.delete_one({"review_id": review_id, "user_id": user["user_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=403, detail="Non puoi eliminare questa recensione")
    return {"ok": True}

@api_router.delete("/movies/{tmdb_id}/public-reviews/{review_id}/reply/{reply_id}")
@limiter.limit("20/minute")
async def delete_reply(request: Request, tmdb_id: int, review_id: str, reply_id: str, user: dict = Depends(get_current_user)):
    result = await db.public_reviews.update_one(
        {"review_id": review_id, "user_id": {"$ne": None}}, 
        {"$pull": {"replies": {"reply_id": reply_id, "user_id": user["user_id"]}}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=403, detail="Non puoi eliminare questa risposta")
    return {"ok": True}

@api_router.get("/movies/{tmdb_id}/public-reviews")
@limiter.limit("60/minute")
async def get_public_reviews(request: Request, tmdb_id: int):
    cursor = db.public_reviews.find({"tmdb_id": tmdb_id}).sort("created_at", -1).limit(50)
    out = []
    async for r in cursor:
        is_anon = r.get("is_anonymous", False)
        likes = r.get("likes", []) # Peschiamo i likes dal database
        out.append({
            "review_id": r["review_id"],
            "user_id": r.get("user_id"),
            "text": r.get("text"),
            "rating": r.get("rating"),
            "created_at": r["created_at"].isoformat() if "created_at" in r else None,
            "user_name": "Utente Anonimo" if is_anon else r.get("user_name", "Utente"),
            "user_picture": None if is_anon else r.get("user_picture"),
            "is_anonymous": is_anon,
            "replies": r.get("replies", []),
            "likes": likes,
            "likes_count": len(likes)
        })
    return {"reviews": out}

@api_router.post("/movies/{tmdb_id}/public-reviews/{review_id}/like")
@limiter.limit("30/minute")
async def toggle_review_like(request: Request, tmdb_id: int, review_id: str, user: dict = Depends(get_current_user)):
    if user.get("auth_provider") == "guest":
        raise HTTPException(status_code=403, detail="Crea un account per mettere like")
    
    uid = user["user_id"]
    review = await db.public_reviews.find_one({"review_id": review_id})
    if not review:
        raise HTTPException(status_code=404, detail="Recensione non trovata")
    
    likes = review.get("likes", [])
    
    # Se ha già messo like, lo togliamo. Se non l'ha messo, lo aggiungiamo.
    if uid in likes:
        likes.remove(uid)
        status = "unliked"
    else:
        likes.append(uid)
        status = "liked"
        
    await db.public_reviews.update_one(
        {"review_id": review_id},
        {"$set": {"likes": likes}}
    )
    return {"ok": True, "status": status, "likes_count": len(likes)}

# ==========================================
# ROTTA PER IL QUIZ
# ==========================================
@api_router.get("/quiz/{tmdb_id}")
@limiter.limit("20/minute")
async def movie_quiz(request: Request, tmdb_id: int):
    if tmdb_id <= 0 or tmdb_id > 10_000_000:
        raise HTTPException(status_code=400, detail="tmdb_id non valido")
    
    # Controlla se il quiz è già salvato nel database
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
    
    # Se la trama in italiano è troppo corta, prova a prenderla in inglese
    if not ctx.get("overview") or len(ctx["overview"]) < 30:
        try:
            en_details = await tmdb_get(f"/movie/{tmdb_id}", {"language": "en-US"})
            ctx["overview"] = (en_details.get("overview") or "").strip()
        except Exception:
            pass
            
    # Se non c'è proprio trama, restituisce un quiz vuoto per non far crashare l'app
    if not ctx.get("overview") or len(ctx["overview"]) < 30:
        return _empty_quiz_payload(ctx, tmdb_id)

    # Chiama Gemini per generare il quiz
    raw = await _call_llm_for_quiz(tmdb_id, _build_quiz_prompt(ctx))
    parsed = _parse_llm_json(raw)
    payload = _normalize_quiz_payload(parsed, ctx, tmdb_id)
    
    # Salva il quiz per la prossima volta
    await _cache_quiz(tmdb_id, payload)
    return payload

@app.get("/")
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

# ==========================================
# SEZIONE NEWS (EDICOLA)
# ==========================================

async def background_news_syncer():
    """Motore che gira in background e aggiorna le news ogni 2 ore"""
    while True:
        try:
            await asyncio.sleep(10) # Partenza ritardata all'avvio
            logger.info("Ricerca nuove notizie di cinema...")
            now = datetime.now(timezone.utc)
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                # 1. PESCA I FILM IN USCITA (TMDB)
                try:
                    up = await tmdb_get("/movie/upcoming", {"region": "IT", "page": 1})
                    for m in up.get("results", [])[:10]:
                        if not m.get("release_date"): continue
                        nid = f"tmdb_{m['id']}"
                        doc = {
                            "news_id": nid,
                            "type": "upcoming",
                            "title": f"🎬 Prossimamente: {m['title']}",
                            "summary": m.get("overview") or "Scopri i dettagli di questa nuova uscita al cinema.",
                            "image_url": f"{TMDB_IMG}{m['backdrop_path']}" if m.get("backdrop_path") else None,
                            "source": "TMDB",
                            "url": None, 
                            "target_id": str(m['id']),
                            "published_at": m.get("release_date")
                        }
                        await db.news_feed.update_one(
                            {"news_id": nid}, 
                            {"$setOnInsert": {**doc, "reactions": {"heart": [], "fire": [], "thumb": []}, "created_at": now}}, 
                            upsert=True
                        )
                except Exception as e:
                    logger.error(f"Errore sincronizzazione TMDB Upcoming: {e}")

                # 2. PESCA LE NOTIZIE VERE E PROPRIE (RSS ANSA CINEMA)
                try:
                    rss_url = "https://www.ansa.it/sito/notizie/cultura/cinema/cinema_rss.xml"
                    
                    # TRUCCO ANTI-BOT: Fingiamo di essere un normale browser Chrome!
                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
                    
                    resp = await client.get(rss_url, headers=headers)
                    if resp.status_code == 200:
                        # Importazione sicura per evitare crash
                        import xml.etree.ElementTree as ET 
                        root = ET.fromstring(resp.text)
                        
                        for item in root.findall(".//item")[:10]:
                            link = item.findtext("link")
                            if not link: continue
                            
                            nid = f"rss_{uuid.uuid5(uuid.NAMESPACE_URL, link).hex}"
                            desc = item.findtext("description")
                            if desc: 
                                desc = desc.replace("<br/>", "\n").replace("<br>", "\n")
                            
                            doc = {
                                "news_id": nid,
                                "type": "article",
                                "title": item.findtext("title"),
                                "summary": desc[:200] + "..." if desc and len(desc) > 200 else desc,
                                "image_url": None, 
                                "source": "ANSA Cinema",
                                "url": link, 
                                "target_id": None,
                                "published_at": now.isoformat()
                            }
                            await db.news_feed.update_one(
                                {"news_id": nid}, 
                                {"$setOnInsert": {**doc, "reactions": {"heart": [], "fire": [], "thumb": []}, "created_at": now}}, 
                                upsert=True
                            )
                    else:
                        logger.error(f"ANSA ha rifiutato la connessione: {resp.status_code}")
                except Exception as e:
                    logger.error(f"Errore sincronizzazione RSS: {e}")

        except Exception as e:
            logger.error(f"Errore generale loop news: {e}")
        
        # Attende 2 ore prima di cercare nuove notizie
        await asyncio.sleep(7200)

@api_router.get("/news")
@limiter.limit("30/minute")
async def get_news_feed(request: Request, user: dict = Depends(get_current_user)):
    cursor = db.news_feed.find({}, {"_id": 0}).sort("created_at", -1).limit(30)
    items = []
    
    async for doc in cursor:
        # TRUCCO DATA: Trasformiamo la data di sistema in testo leggibile dall'App
        if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
            doc["created_at"] = doc["created_at"].isoformat()
        items.append(doc)
        
    return {"items": items}

# ==========================================
# MOTORE NOTIFICHE ATTORI IN BACKGROUND
# ==========================================
async def background_actor_checker():
    while True:
        try:
            # Aspetta 1 minuto all'avvio del server prima di iniziare
            await asyncio.sleep(60) 
            logger.info("Inizio controllo giornaliero nuovi film attori...")
            
            # 1. Prendi tutti gli ID degli attori che sono seguiti da almeno un utente
            followed_people = await db.user_people.distinct("person_id")
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
            async with httpx.AsyncClient() as client:
                for person_id in followed_people:
                    try:
                        # 2. Chiedi a TMDB i film di questo attore
                        credits = await tmdb_get(f"/person/{person_id}/movie_credits", {"language": "it-IT"})
                        cast = credits.get("cast", [])
                        
                        for movie in cast:
                            release_date = movie.get("release_date")
                            movie_id = movie.get("id")
                            
                            # Se il film non ha una data, saltalo
                            if not release_date:
                                continue

                            # Controlliamo se lo abbiamo già notificato in passato per non spammare
                            already_notified = await db.notifications_log.find_one({"person_id": person_id, "movie_id": movie_id})
                            
                            # 3. Se NON lo abbiamo mai notificato E la data di uscita è futura o recente (esce da oggi in poi)
                            if not already_notified and release_date >= today_str:
                                # Segna nel database che lo stiamo notificando oggi
                                await db.notifications_log.insert_one({
                                    "person_id": person_id, 
                                    "movie_id": movie_id, 
                                    "created_at": datetime.now(timezone.utc)
                                })
                                
                                # 4. Trova TUTTI gli utenti che seguono questo attore
                                followers = db.user_people.find({"person_id": person_id})
                                async for f in followers:
                                    user_data = await db.users.find_one({"user_id": f["user_id"]})
                                    if user_data and user_data.get("push_token"):
                                        token = user_data["push_token"]
                                        actor_name = f.get("name", "Un attore che segui")
                                        movie_title = movie.get("title", "un nuovo film")
                                        
                                        # 5. Prepara e invia la notifica PUSH al telefono (USANDO IL NUOVO HELPER!)
                                        await send_expo_push(
                                            client,
                                            token=token,
                                            title=f"🎬 Novità per {actor_name}!",
                                            body=f"È in arrivo «{movie_title}» — Uscita: {release_date}",
                                            data={
                                                "type": "actor_new_movie",
                                                "tmdb_id": movie_id,
                                                "person_id": person_id,
                                                "actor_name": actor_name,
                                                "release_date": release_date,
                                                "screen": "movie",
                                            },
                                        )
                                            
                    except Exception as e:
                        logger.error(f"Errore recupero crediti per attore {person_id}: {e}")
                        
                    # Piccola pausa tra un attore e l'altro per non far bloccare TMDB (Rate Limit)
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Errore nel loop notifiche: {e}")
            
        # Ripeti il controllo ogni 24 ore (86400 secondi)
        await asyncio.sleep(86400)

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
    await db.public_reviews.create_index([("tmdb_id", 1), ("created_at", -1)])
    await db.user_people.create_index([("user_id", 1), ("person_id", 1)], unique=True)
    
    # --- NUOVO: Indice per non inviare doppie notifiche ---
    await db.notifications_log.create_index([("person_id", 1), ("movie_id", 1)], unique=True)
    
    # --- NUOVO: Il "netturbino" di MongoDB che cancella i dati TMDB vecchi di 48 ore ---
    await db.tmdb_cache.create_index([("updated_at", 1)], expireAfterSeconds=172800)
    
    # --- NUOVO: Avvia il motore delle notifiche in background ---
    asyncio.create_task(background_actor_checker())
    # --- NUOVO: Avvia il motore che scarica le news ---
    asyncio.create_task(background_news_syncer())
    
    logger.info("CineDiario backend ready")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
