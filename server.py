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
from typing import Optional
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

COUNTRY_IT = {
    "United States of America": "Stati Uniti",
    "United Kingdom": "Regno Unito",
    "France": "Francia",
    "Germany": "Germania",
    "Italy": "Italia",
    "Spain": "Spagna",
    "Japan": "Giappone",
    "China": "Cina",
    "South Korea": "Corea del Sud",
    "Korea, South": "Corea del Sud",
    "India": "India",
    "Canada": "Canada",
    "Australia": "Australia",
    "New Zealand": "Nuova Zelanda",
    "Brazil": "Brasile",
    "Argentina": "Argentina",
    "Mexico": "Messico",
    "Russia": "Russia",
    "Soviet Union": "Unione Sovietica",
    "Sweden": "Svezia",
    "Norway": "Norvegia",
    "Denmark": "Danimarca",
    "Finland": "Finlandia",
    "Iceland": "Islanda",
    "Belgium": "Belgio",
    "Netherlands": "Paesi Bassi",
    "Switzerland": "Svizzera",
    "Austria": "Austria",
    "Ireland": "Irlanda",
    "Portugal": "Portogallo",
    "Greece": "Grecia",
    "Turkey": "Turchia",
    "Poland": "Polonia",
    "Czech Republic": "Repubblica Ceca",
    "Hungary": "Ungheria",
    "Romania": "Romania",
    "Bulgaria": "Bulgaria",
    "Israel": "Israele",
    "Iran": "Iran",
    "Egypt": "Egitto",
    "South Africa": "Sudafrica",
    "Hong Kong": "Hong Kong",
    "Taiwan": "Taiwan",
    "Thailand": "Thailandia",
    "Vietnam": "Vietnam",
    "Indonesia": "Indonesia",
    "Philippines": "Filippine",
    "Malaysia": "Malesia",
    "Singapore": "Singapore",
    "United Arab Emirates": "Emirati Arabi Uniti",
    "Saudi Arabia": "Arabia Saudita",
    "Chile": "Cile",
    "Colombia": "Colombia",
    "Peru": "Perù",
    "Cuba": "Cuba",
    "Ukraine": "Ucraina",
    "Croatia": "Croazia",
    "Serbia": "Serbia",
    "Yugoslavia": "Jugoslavia",
    "West Germany": "Germania Ovest",
    "East Germany": "Germania Est",
}


def translate_country(name: str) -> str:
    if not name:
        return name
    return COUNTRY_IT.get(name, name)


# ===== Rate limiter =====
def _rate_key(request: Request) -> str:
    """Rate limit per token se autenticato, altrimenti per IP."""
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
        response.headers.pop("Server", None)
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Payload troppo grande"})
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
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": {"friend_code": fallback}},
    )
    user["friend_code"] = fallback
    return fallback


def _extract_bearer_token(authorization: Optional[str], client_ip: str) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        security_logger.warning(f"AUTH_FAIL missing_token ip={client_ip}")
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization[7:].strip()
    if not token or len(token) < 20:
        security_logger.warning(f"AUTH_FAIL malformed_token ip={client_ip}")
        raise HTTPException(status_code=401, detail="Invalid token")
    return token


def _verify_firebase_token(id_token: str, client_ip: str) -> dict:
    try:
        decoded = firebase_auth.verify_id_token(id_token, check_revoked=True)
    except firebase_auth.RevokedIdTokenError:
        security_logger.warning(f"AUTH_FAIL revoked_token ip={client_ip}")
        raise HTTPException(status_code=401, detail="Token revoked")
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Token expired")
    except firebase_auth.UserDisabledError:
        security_logger.warning(f"AUTH_FAIL user_disabled ip={client_ip}")
        raise HTTPException(status_code=401, detail="User disabled")
    except firebase_auth.InvalidIdTokenError as e:
        security_logger.warning(f"AUTH_FAIL invalid_token ip={client_ip} reason={e}")
        raise HTTPException(status_code=401, detail="Invalid Firebase token")
    except Exception as e:
        security_logger.error(f"AUTH_FAIL verify_error ip={client_ip} err={e}")
        raise HTTPException(status_code=401, detail="Token verification error")

    expected_iss = f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}"
    if decoded.get("aud") != FIREBASE_PROJECT_ID or decoded.get("iss") != expected_iss:
        security_logger.warning(
            f"AUTH_FAIL wrong_project ip={client_ip} aud={decoded.get('aud')} iss={decoded.get('iss')}"
        )
        raise HTTPException(status_code=401, detail="Token not for this project")
    return decoded


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


async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> dict:
    """Orchestratore auth: estrae token → verifica Firebase → upsert utente."""
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
    return round(sum(filled) / len(filled), 1)


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


# ===== Health =====
@api_router.get("/health")
async def health():
    try:
        await db.command("ping")
        mongo_ok = True
    except Exception:
        mongo_ok = False
    return {"status": "ok", "mongo": mongo_ok, "firebase": bool(firebase_admin._apps)}


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
            "known_for": [
                k.get("title") or k.get("name")
                for k in p.get("known_for", [])[:3]
                if k.get("title") or k.get("name")
            ],
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
            "id": c["id"],
            "name": c.get("name"),
            "character": c.get("character"),
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
        "keywords": [
            k.get("name")
            for k in (details.get("keywords", {}) or {}).get("keywords", [])[:8]
            if k.get("name")
        ],
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
    """Chiama direttamente l'API REST di Google Gemini (free tier)."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="LLM non configurato")
    system_msg = (
        "Sei un esperto di cinema italiano. Generi quiz a scelta multipla sulla TRAMA dei film. "
        "Rispondi SOLO con un oggetto JSON valido, senza markdown, senza testo prima o dopo."
    )
    body = {
        "systemInstruction": {"parts": [{"text": system_msg}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "responseMimeType": "application/json",
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(
                GEMINI_URL,
                params={"key": GEMINI_API_KEY},
                json=body,
            )
        if r.status_code != 200:
            logger.error(f"Gemini API error tmdb_id={tmdb_id} status={r.status_code} body={r.text[:300]}")
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
        logger.error(f"Gemini quiz generation failed for tmdb_id={tmdb_id}: {e}")
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
        "tmdb_id": tmdb_id,
        "title": ctx["title"],
        "poster_url": ctx["poster_url"],
        "tagline": ctx["tagline"],
        "questions": [],
        "recap": {
            "intro": f"Trama di «{ctx['title']}» non disponibile in italiano su TMDB.",
            "plot": "",
            "characters": [],
            "key_moments": [],
            "themes": [],
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
                "tmdb_id": tmdb_id,
                "lang": "it",
                "version": 2,
                "payload": payload,
                "generated_at": datetime.now(timezone.utc),
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

# ===== Middleware (DOPO include_router) =====
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
    logger.info("CineDiario backend ready (Firebase Auth + MongoDB Atlas + Gemini)")


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
