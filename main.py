import asyncio
import base64
import hashlib
import json
import os
import random
import time
import traceback
import urllib.parse
import uuid

import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from google import genai
from google.genai import types
from pydantic import BaseModel
from cutout import make_cutout_data_uri

app = FastAPI(title="StoryDoodle API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👇 Anahtarını tırnakların içine yapıştır (dosyayı GitHub'a yükleme!)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=90000),  # 90 sn
)

MODELS_TO_TRY = [
    "gemini-3.5-flash-lite",  # genelde daha az yoğun, önce bunu dene
    "gemini-3.6-flash",
    "gemini-2.5-flash",       # son çare; hesapta yoksa log'da hata yazar, zararı olmaz
]

# Her hikayede farklı bir isim havası seçilir, böylece hep aynı tarz isimler çıkmaz
NAME_THEMES_TR = [
    "tatlı ve yiyecek temalı komik isimler (ör. Tarçın, Şeker Bulut, Pofuduk Börek gibi ruhta)",
    "unvanlı eğlenceli isimler (Kaptan, Profesör, Şövalye, Sultan, Dedektif ile başlayan)",
    "ses taklidi tekrarlı isimler (Pıtırcık, Zıpzıp, Mırmır, Vıcık gibi ruhta)",
    "doğa ve hava olayı temalı isimler (Çiy, Yıldırım Tozu, Kar Tanesi, Rüzgar gibi ruhta)",
    "büyülü ve masalsı isimler (Işıldak, Sihirli Fındık, Gökkuşağı Kaşifi gibi ruhta)",
    "iki kelimeli tuhaf isimler (Bulut Horozu, Limon Şövalyesi, Patates Prens gibi ruhta)",
]
NAME_THEMES_EN = [
    "sweet food-themed funny names (like Cinnamon Bun, Sugar Cloud)",
    "titled playful names (Captain, Professor, Sir, Detective ...)",
    "bouncy repeated-sound names (like Zippity, Boop-Boop, Wiggly)",
    "nature and weather themed names (like Dewdrop, Thunder Fluff)",
    "magical fairytale names (like Glimmerwick, Rainbow Ranger)",
    "quirky two-word names (like Cloud Rooster, Lemon Knight)",
]
FALLBACK_NAMES_TR = ["Tarçın", "Kaptan Pıtırcık", "Zıpzıp", "Bulut Şekeri", "Profesör Mırmır", "Işıldak", "Limon Şövalyesi"]
FALLBACK_NAMES_EN = ["Cinnamon", "Captain Zippity", "Boop-Boop", "Cloud Candy", "Professor Wiggly", "Glimmerwick", "Lemon Knight"]

# Kotası dolan modeli bir süre denemeyerek beklemeyi önler
MODEL_COOLDOWN = {}  # model_adi -> tekrar denenebileceği zaman (epoch)
COOLDOWN_SECONDS = 90

CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Uygulamanın (App.tsx) bildiği animasyon tipleri
ALLOWED_ANIMATIONS = {"zipla", "ileri_geri", "sallan", "donme", "nefes_al", "yuz", "duman"}
ALLOWED_OBJECT_TYPES = {"hayvan", "arac", "bina", "bitki", "diger"}


class InitialStoryRequest(BaseModel):
    image_base64: str
    narrator_id: str = "storyteller"
    language: str = "tr"  # "tr" | "en"
    speed: float = 1.0    # 0.8, 1.0, 1.2
    # Çocuk "hazır şablon" kullanarak nokta nokta çizim yaptıysa,
    # animasyonu Gemini'nin tahminine bırakmadan doğrudan bunu kullanırız.
    template_animasyon_tipi: str | None = None
    template_nesne_turu: str | None = None
    remove_background: bool = False


class ContinueStoryRequest(BaseModel):
    karakter_adi: str
    karakter_tanimi: str = ""
    onceki_metin: str = ""
    secilen_yol: str
    bolum_sayisi: int = 2
    narrator_id: str = "storyteller"
    language: str = "tr"
    speed: float = 1.0


class TTSRequest(BaseModel):
    text: str
    narrator_id: str
    language: str = "tr"
    speed: float = 1.0


@app.get("/")
def home():
    return {"status": "StoryDoodle Multilingual API Çalışıyor! 🚀"}


@app.get("/bg-music")
async def get_background_music():
    music_url = "https://actions.google.com/sounds/v1/science_fiction/lullaby_soundscape.ogg"
    return RedirectResponse(url=music_url)


def clean_json_response(raw_text: str) -> dict:
    t = raw_text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return json.loads(t.strip())


def create_ai_image_url(prompt_desc: str, width: int = 400, height: int = 240) -> str:
    clean_prompt = (
        f"3d cartoon children storybook illustration of {prompt_desc}, vibrant colors, "
        f"Disney Pixar style, cute, fairytale, rich detailed environment, no text"
    )
    encoded = urllib.parse.quote(clean_prompt)
    seed = uuid.uuid4().int % 99999
    # DİKKAT: düz URL olmalı (markdown köşeli parantez yok)
    return (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&nologo=true&seed={seed}&model=flux"
    )


def normalize_animation_fields(data: dict) -> dict:
    """Gemini'den gelen animasyon alanlarını doğrular; hatalıysa güvenli varsayılan verir."""
    anim = data.get("animasyon_tipi")
    if not isinstance(anim, str) or anim not in ALLOWED_ANIMATIONS:
        anim = "nefes_al"
    data["animasyon_tipi"] = anim

    obj = data.get("nesne_turu")
    if not isinstance(obj, str) or obj not in ALLOWED_OBJECT_TYPES:
        obj = "diger"
    data["nesne_turu"] = obj

    point = data.get("efekt_noktasi")
    clean_point = None
    if isinstance(point, dict):
        try:
            x = float(point.get("x"))
            y = float(point.get("y"))
            clean_point = {"x": min(max(x, 0.0), 1.0), "y": min(max(y, 0.0), 1.0)}
        except (TypeError, ValueError):
            clean_point = None

    # Duman animasyonu için nokta şart; yoksa makul bir varsayılan (sağ üst)
    if anim == "duman" and clean_point is None:
        clean_point = {"x": 0.7, "y": 0.25}
    data["efekt_noktasi"] = clean_point
    return data


def generate_with_fallback(contents, config=None):
    now = time.time()
    for model_name in MODELS_TO_TRY:
        if MODEL_COOLDOWN.get(model_name, 0) > now:
            print(f"~ {model_name} soğuma sürecinde, atlanıyor")
            continue

        # Google tarafında geçici yoğunluk (503) sık görülüyor; aynı modeli
        # artan bekleme süreleriyle birkaç kez daha deneriz, hemen pes etmeyiz.
        backoffs = [0, 1.5, 3.5]
        for attempt, wait_s in enumerate(backoffs):
            if wait_s:
                time.sleep(wait_s)
            try:
                print(f"--> Gemini deneniyor: {model_name} (deneme {attempt + 1}/{len(backoffs)})")
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                print(f"✓ Başarılı model: {model_name}")
                return response.text
            except Exception as e:
                msg = str(e)
                print(f"x {model_name} başarısız: {msg[:300]}")

                # Kota doldu: bu modeli bir süre dinlendir, hemen sıradakine geç
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    MODEL_COOLDOWN[model_name] = time.time() + COOLDOWN_SECONDS
                    break

                # 503/504 (yoğunluk/zaman aşımı): backoff listesindeki süreyle tekrar dene
                if "503" in msg or "UNAVAILABLE" in msg or "504" in msg or "DEADLINE" in msg:
                    continue

                # Başka bir hata: bu modelde ısrar etmenin anlamı yok, sıradakine geç
                break

    print("⚠️ Tüm modeller başarısız, yedek hikaye motoru devrede.")
    return None


def get_audio_cache_key(text: str, narrator_id: str, language: str, speed: float) -> str:
    raw = f"{language}:{narrator_id}:{speed}:{text.strip()}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


async def generate_audio_base64(text: str, narrator_id: str, language: str = "tr", speed: float = 1.0) -> str:
    if not text.strip():
        return ""

    cache_key = get_audio_cache_key(text, narrator_id, language, speed)
    cached_path = os.path.join(CACHE_DIR, f"{cache_key}.mp3")

    if os.path.exists(cached_path):
        with open(cached_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Ses & Ton Seçimi (Türkçe vs İngilizce)
    if language == "en":
        voice = "en-US-GuyNeural"
        base_rate = 0
        pitch = "-2Hz"

        if narrator_id == "fairy":
            voice, pitch = "en-US-AnaNeural", "+4Hz"
        elif narrator_id == "chipmunk":
            voice, pitch = "en-US-AnaNeural", "+40Hz"
            base_rate += 25
        elif narrator_id == "dragon":
            voice, pitch = "en-US-ChristopherNeural", "-25Hz"
            base_rate -= 15
        elif narrator_id == "robot":
            voice, pitch = "en-US-EricNeural", "+18Hz"
        elif narrator_id == "wise":
            voice, pitch = "en-US-RogerNeural", "-12Hz"
    else:
        voice = "tr-TR-AhmetNeural"
        base_rate = -8
        pitch = "-4Hz"

        if narrator_id == "dragon":
            base_rate, pitch = -18, "-30Hz"
        elif narrator_id == "wise":
            base_rate, pitch = -12, "-10Hz"
        elif narrator_id == "robot":
            base_rate, pitch = +16, "+20Hz"
        elif narrator_id == "chipmunk":
            voice, base_rate, pitch = "tr-TR-EmelNeural", +30, "+45Hz"
        elif narrator_id == "fairy":
            voice, base_rate, pitch = "tr-TR-EmelNeural", -6, "+3Hz"

    # Kullanıcının hız ayarını (0.8x, 1.0x, 1.2x) ekle
    speed_offset = int((speed - 1.0) * 100)
    final_rate_int = base_rate + speed_offset
    rate_str = f"{'+' if final_rate_int >= 0 else ''}{final_rate_int}%"

    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch)
        await communicate.save(cached_path)

        with open(cached_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Edge-TTS Error: {e}")
        return ""


ANIMATION_RULES_EN = """
            5. Pick the animation that best fits the MAIN object in the drawing:
               - "animasyon_tipi": one of "zipla" (hopping animals: rabbit, frog, kangaroo, ball),
                 "ileri_geri" (vehicles: car, bus, train, boat, plane),
                 "sallan" (plants/swaying things: flower, tree, balloon),
                 "donme" (sun, wheel, star, flower head),
                 "yuz" (swimming/floating: fish, whale, duck),
                 "duman" (buildings with a chimney: house, factory, castle),
                 "nefes_al" (anything else: cat, person, dog, robot...).
               - "efekt_noktasi": {"x": 0-1, "y": 0-1} position (fraction of image width/height, origin top-left)
                 where smoke should come out (the chimney top). Only for "duman", otherwise null.
               - "nesne_turu": one of "hayvan", "arac", "bina", "bitki", "diger".
"""

ANIMATION_RULES_TR = """
            5. Çizimdeki ANA nesneye en uygun animasyonu seç:
               - "animasyon_tipi": şunlardan biri:
                 "zipla" (zıplayan canlılar: tavşan, kurbağa, kanguru, top),
                 "ileri_geri" (araçlar: araba, otobüs, tren, gemi, uçak),
                 "sallan" (bitkiler/sallanan şeyler: çiçek, ağaç, balon),
                 "donme" (güneş, tekerlek, yıldız),
                 "yuz" (yüzen/süzülen: balık, balina, ördek),
                 "duman" (bacası olan yapılar: ev, fabrika, kale),
                 "nefes_al" (diğer her şey: kedi, insan, köpek, robot...).
               - "efekt_noktasi": {"x": 0-1, "y": 0-1} dumanın çıkacağı nokta (görselin genişlik/yükseklik oranı,
                 sol-üst köşe 0,0). Sadece "duman" için doldur, diğerlerinde null.
               - "nesne_turu": "hayvan", "arac", "bina", "bitki" veya "diger".
"""


@app.post("/generate-story")
async def generate_story(req: InitialStoryRequest):
    try:
        image_bytes = base64.b64decode(req.image_base64)
        is_en = req.language == "en"
        name_theme = random.choice(NAME_THEMES_EN if is_en else NAME_THEMES_TR)

        if is_en:
            prompt = f"""
            You are a world-renowned children's storybook author.
            Carefully inspect the child's drawing:
            1. Identify the main character and its cute details. Give it a CREATIVE, memorable, playful name. Name style for this story: {name_theme}. Avoid generic names like Fluffy, Buddy or Pofu; be original.
            2. Write Chapter 1 of the fairytale in ENGLISH (approx 70-85 words, cheerful, curious, engaging). The hero arrives at a magical crossroads.
            3. Provide a 4-6 word English prompt describing the main scene of Chapter 1 (sahne_img_prompt).
            4. Offer 2 short exciting path choices in ENGLISH for the child:
               - An emoji for each choice (secenek_1_emoji, secenek_2_emoji),
               - A 4-6 word English scene prompt for each path (secenek_1_img_prompt, secenek_2_img_prompt).
            {ANIMATION_RULES_EN}

            Output strictly JSON:
            {{
              "karakter_adi": "...",
              "karakter_tanimi": "...",
              "masal_basligi": "...",
              "bolum_metni": "...",
              "sahne_img_prompt": "cute hero exploring a magical enchanted meadow with glowing flowers",
              "secenek_1": "...",
              "secenek_1_emoji": "🌲",
              "secenek_1_img_prompt": "magical glowing mushroom forest pathway",
              "secenek_2": "...",
              "secenek_2_emoji": "🎈",
              "secenek_2_img_prompt": "colorful hot air balloon over rainbow hills",
              "animasyon_tipi": "zipla",
              "efekt_noktasi": null,
              "nesne_turu": "hayvan",
              "is_final": false
            }}
            """
        else:
            prompt = f"""
            Sen dünyaca ünlü bir çocuk masalları yazarısın.
            Görseldeki çocuk çizimini dikkatle incele:
            1. Çizimdeki ana karakteri ve detaylarını keşfet, YARATICI, akılda kalıcı, eğlenceli bir isim koy. Bu masal için isim havası: {name_theme}. Pofu, Minik, Tatlı gibi sık kullanılan sıradan isimlerden kaçın; özgün ol.
            2. Masalın 1. BÖLÜMÜNÜ TÜRKÇE yaz (ortalama 70-85 kelime, neşeli, merak uyandırıcı). Karakter iki farklı büyülü yolun başına gelsin.
            3. Bu 1. bölümün ana sahnesini betimleyen İngilizce bir görsel istemi yaz (sahne_img_prompt).
            4. Çocuğun seçeceği 2 farklı yol ayrımı belirle:
               - Her seçenek için uygun bir emoji (secenek_1_emoji, secenek_2_emoji),
               - O yolun görselini çizecek kısa İngilizce prompt (secenek_1_img_prompt, secenek_2_img_prompt).
            {ANIMATION_RULES_TR}

            Çıktıyı doğrudan şu JSON formatında ver:
            {{
              "karakter_adi": "...",
              "karakter_tanimi": "...",
              "masal_basligi": "...",
              "bolum_metni": "...",
              "sahne_img_prompt": "cute hero exploring a magical enchanted meadow with glowing flowers",
              "secenek_1": "...",
              "secenek_1_emoji": "🌲",
              "secenek_1_img_prompt": "magical glowing mushroom forest pathway",
              "secenek_2": "...",
              "secenek_2_emoji": "🎈",
              "secenek_2_img_prompt": "colorful hot air balloon over rainbow hills",
              "animasyon_tipi": "zipla",
              "efekt_noktasi": null,
              "nesne_turu": "hayvan",
              "is_final": false
            }}
            """

        raw_text = await asyncio.to_thread(
            generate_with_fallback,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        if raw_text:
            data = clean_json_response(raw_text)
        else:
            if is_en:
                data = {
                    "karakter_adi": "Fluffy",
                    "karakter_tanimi": "Curious little woodland explorer",
                    "masal_basligi": "The Secret of the Magic Forest",
                    "bolum_metni": "As the warm golden sunshine filtered through the emerald leaves, little Fluffy was hopping with joy! Today felt like a special day in the forest; birds were singing happy songs, and sweet vanilla scents drifted through the air. Walking down the flower-lined path, Fluffy suddenly discovered a magnificent crossroads! On the left rose a glittering crystal mountain, while on the right, giant rainbow balloons were floating above the clouds.",
                    "sahne_img_prompt": "cute furry creature standing at a whimsical crossroads in a fairytale enchanted forest",
                    "secenek_1": "Climb the Sparkling Crystal Mountain",
                    "secenek_1_emoji": "💎",
                    "secenek_1_img_prompt": "sparkling crystal mountain fantasy pathway",
                    "secenek_2": "Fly With the Giant Rainbow Balloons",
                    "secenek_2_emoji": "🎈",
                    "secenek_2_img_prompt": "giant colorful hot air balloons floating over clouds",
                    "animasyon_tipi": "zipla",
                    "efekt_noktasi": None,
                    "nesne_turu": "hayvan",
                    "is_final": False,
                }
            else:
                data = {
                    "karakter_adi": "Pofu",
                    "karakter_tanimi": "Meraklı orman kaşifi",
                    "masal_basligi": "Sihirli Ormanın Büyük Gizemi",
                    "bolum_metni": "Güneşin altın sarısı ışıkları yemyeşil yaprakların arasından süzülürken, sevimli kahramanımız Pofu neşeyle yerinde zıplıyordu. Bugün ormanda çok özel bir gün gibiydi; kuşlar neşeli şarkılar söylüyor, rüzgar tatlı bir vanilya kokusu getiriyordu. Pofu minik adımlarla çiçekli patikada ilerlerken birden karşısına devasa, sihirli bir yol ayrımı çıktı! Sol tarafta gökyüzüne kadar uzanan pırıltılı kristal bir dağ, sağ tarafta ise bulutların üstünde süzülen dev gökkuşağı balonları onu bekliyordu.",
                    "sahne_img_prompt": "cute furry creature standing at a whimsical crossroads in a fairytale enchanted forest",
                    "secenek_1": "Pırıltılı Kristal Dağa Tırman",
                    "secenek_1_emoji": "💎",
                    "secenek_1_img_prompt": "sparkling crystal mountain fantasy pathway",
                    "secenek_2": "Uçan Gökkuşağı Balonlarına Koş",
                    "secenek_2_emoji": "🎈",
                    "secenek_2_img_prompt": "giant colorful hot air balloons floating over clouds",
                    "animasyon_tipi": "zipla",
                    "efekt_noktasi": None,
                    "nesne_turu": "hayvan",
                    "is_final": False,
                }

        if not raw_text:
            fb_name = random.choice(FALLBACK_NAMES_EN if is_en else FALLBACK_NAMES_TR)
            old_name = "Fluffy" if is_en else "Pofu"
            data["bolum_metni"] = data["bolum_metni"].replace(old_name, fb_name)
            data["karakter_adi"] = fb_name

        # Animasyon alanlarını doğrula / varsayılanla
        data = normalize_animation_fields(data)

        # Şablon kullanıldıysa (nokta nokta), animasyonu kesin bilgiyle override et
        if req.template_animasyon_tipi and req.template_animasyon_tipi in ALLOWED_ANIMATIONS:
            data["animasyon_tipi"] = req.template_animasyon_tipi
        if req.template_nesne_turu and req.template_nesne_turu in ALLOWED_OBJECT_TYPES:
            data["nesne_turu"] = req.template_nesne_turu

        scene_prompt = data.get("sahne_img_prompt", f"cute {data.get('karakter_adi', 'hero')} in fairytale wonderland")
        data["sahne_img_url"] = create_ai_image_url(scene_prompt, width=480, height=260)

        p1 = data.get("secenek_1_img_prompt", "magical fantasy road")
        p2 = data.get("secenek_2_img_prompt", "whimsical rainbow pathway")
        data["secenek_1_img_url"] = create_ai_image_url(p1, width=300, height=300)
        data["secenek_2_img_url"] = create_ai_image_url(p2, width=300, height=300)

        if "secenek_1_emoji" not in data:
            data["secenek_1_emoji"] = "✨"
        if "secenek_2_emoji" not in data:
            data["secenek_2_emoji"] = "🌟"

        data["cutout_image"] = None
        if req.remove_background:
            try:
                data["cutout_image"] = await asyncio.to_thread(make_cutout_data_uri, image_bytes)
            except Exception:
                traceback.print_exc()

        data["audio_base64"] = ""
        return data
    except Exception as e:
        print("\n--- HATA: /generate-story ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/continue-story")
async def continue_story(req: ContinueStoryRequest):
    try:
        is_last_part = req.bolum_sayisi >= 3
        is_en = req.language == "en"

        if is_en:
            prompt = f"""
            You are a charming children's storybook author.
            Hero: {req.karakter_adi}
            Story so far: "{req.onceki_metin}"
            Chosen path: "{req.secilen_yol}".

            TASK:
            1. Write the new chapter in ENGLISH based on the choice (approx 70-85 words).
            2. Describe the scene in English (sahne_img_prompt).
            3. Final chapter? {"YES! Give it a joyful, warm and happy ending." if is_last_part else "NO! Adventure continues, end with 2 new choices."}
            4. If not final, offer 2 new choices in ENGLISH with emojis and image prompts. If final, leave choices empty.

            Output strictly JSON:
            {{
              "karakter_adi": "{req.karakter_adi}",
              "karakter_tanimi": "{req.karakter_tanimi}",
              "masal_basligi": "The Journey Continues",
              "bolum_metni": "...",
              "sahne_img_prompt": "...",
              "secenek_1": "...",
              "secenek_1_emoji": "...",
              "secenek_1_img_prompt": "...",
              "secenek_2": "...",
              "secenek_2_emoji": "...",
              "secenek_2_img_prompt": "...",
              "is_final": {"true" if is_last_part else "false"}
            }}
            """
        else:
            prompt = f"""
            Sen büyüleyici çocuk masalları anlatan usta bir yazarsın.
            Karakter: {req.karakter_adi}
            Önceki yaşananlar: "{req.onceki_metin}"
            Çocuğun seçtiği yeni yol: "{req.secilen_yol}".

            GÖREVİN:
            1. Çocuğun yaptığı seçime göre yeni bölümü TÜRKÇE yaz (ortalama 70-85 kelime).
            2. Bu yeni bölümün can alıcı sahnesini betimleyen İngilizce görsel prompt'u yaz (sahne_img_prompt).
            3. Son Bölüm mü (is_final):
               {"EVET! Masalı sıcacık, mutlu bir sonla bitir." if is_last_part else "HAYIR! Macera sürüyor, karakteri iki yeni seçeneğin önüne getir."}
            4. Son bölüm değilse 2 yeni seçenek, emoji ve görsel prompt'ları üret. Son bölümse seçenekleri boş bırak.

            Çıktıyı doğrudan şu JSON formatında ver:
            {{
              "karakter_adi": "{req.karakter_adi}",
              "karakter_tanimi": "{req.karakter_tanimi}",
              "masal_basligi": "Maceranın Devamı",
              "bolum_metni": "...",
              "sahne_img_prompt": "...",
              "secenek_1": "...",
              "secenek_1_emoji": "...",
              "secenek_1_img_prompt": "...",
              "secenek_2": "...",
              "secenek_2_emoji": "...",
              "secenek_2_img_prompt": "...",
              "is_final": {"true" if is_last_part else "false"}
            }}
            """

        raw_text = await asyncio.to_thread(
            generate_with_fallback,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        if raw_text:
            data = clean_json_response(raw_text)
        else:
            if is_last_part:
                data = {
                    "karakter_adi": req.karakter_adi,
                    "karakter_tanimi": req.karakter_tanimi,
                    "masal_basligi": "Grand Happy Ending!" if is_en else "Görkemli Mutlu Son!",
                    "bolum_metni": (
                        f"Taking the path of {req.secilen_yol}, our brave hero reached a glowing valley where all the lovely woodland creatures had prepared a grand feast of sweets and glowing berries! They sang and danced together under the starry sky, living happily ever after."
                        if is_en else
                        f"{req.secilen_yol} adımını atan cesur kahramanımız, yolun sonunda parıldayan büyük bir vadiye ulaştı. Vadideki tüm sevimli orman canlıları, rengarenk ışıklar ve lezzetli meyvelerle dolu muazzam bir şölen sofrası kurmuştu! Hep birlikte neşeyle şarkılar söylediler, dans ettiler ve kahramanımızın bu cesur yolculuğunu kutladılar. Kahramanımız yeni dostlarıyla birlikte bir ömür boyu mutluluk ve neşe içinde yaşadı."
                    ),
                    "sahne_img_prompt": "grand magical feast party celebration with cute animals fireworks and joyful lights",
                    "secenek_1": "",
                    "secenek_2": "",
                    "secenek_1_emoji": "",
                    "secenek_2_emoji": "",
                    "is_final": True,
                }
            else:
                data = {
                    "karakter_adi": req.karakter_adi,
                    "karakter_tanimi": req.karakter_tanimi,
                    "masal_basligi": "Chasing the Mystery" if is_en else "Büyülü Sırların Peşinde",
                    "bolum_metni": (
                        f"Marching eagerly along {req.secilen_yol}, sparkling stardust fell gently from the clouds. Suddenly, two mysterious gates appeared! One gate led to an ancient castle with golden locks, while the other ascended to a cheerful theme park atop fluffy clouds."
                        if is_en else
                        f"Kahramanımız {req.secilen_yol} yönüne doğru merakla adımlarını hızlandırdı. Çevresindeki ağaçlar tatlı tatlı fısıldıyor, gökyüzünden minik parlak yıldız tozları dökülüyordu. Tam bu sırada önüne ışıl ışıl parlayan iki kapı çıktı! Bir kapı altın anahtarlarla kilitlenmiş antika bir kaleye açılıyor, diğeri ise bulutların üstüne kurulan eğlenceli bir lunaparka uzanıyordu."
                    ),
                    "sahne_img_prompt": "hero walking through sparkling enchanted gateway in wonderland",
                    "secenek_1": "Enter the Golden Castle" if is_en else "Altın Anahtarlı Gizemli Kaleye Gir",
                    "secenek_1_emoji": "🗝️",
                    "secenek_1_img_prompt": "enchanted golden castle door",
                    "secenek_2": "Visit the Cloud Theme Park" if is_en else "Bulutların Üstündeki Lunaparka Git",
                    "secenek_2_emoji": "🎡",
                    "secenek_2_img_prompt": "magical amusement park on clouds",
                    "is_final": False,
                }

        scene_prompt = data.get("sahne_img_prompt", f"{req.secilen_yol} in magical world")
        data["sahne_img_url"] = create_ai_image_url(scene_prompt, width=480, height=260)

        if "secenek_1" not in data:
            data["secenek_1"] = ""
        if "secenek_2" not in data:
            data["secenek_2"] = ""
        if "secenek_1_emoji" not in data:
            data["secenek_1_emoji"] = "🌈"
        if "secenek_2_emoji" not in data:
            data["secenek_2_emoji"] = "⭐"
        if "is_final" not in data:
            data["is_final"] = is_last_part

        if not data["is_final"]:
            p1 = data.get("secenek_1_img_prompt", "fairytale castle pathway")
            p2 = data.get("secenek_2_img_prompt", "magical enchanted river road")
            data["secenek_1_img_url"] = create_ai_image_url(p1, width=300, height=300)
            data["secenek_2_img_url"] = create_ai_image_url(p2, width=300, height=300)
        else:
            data["secenek_1_img_url"] = ""
            data["secenek_2_img_url"] = ""

        data["audio_base64"] = ""
        return data

    except Exception as e:
        print("\n--- HATA: /continue-story ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-voice")
async def generate_voice(req: TTSRequest):
    try:
        audio_base64 = await generate_audio_base64(
            text=req.text,
            narrator_id=req.narrator_id,
            language=req.language,
            speed=req.speed,
        )
        return {"audio_base64": audio_base64}
    except Exception as e:
        print("\n--- HATA: /generate-voice ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
