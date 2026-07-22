# Chat UI — InfinityFree වලින් Host කරන්න

සරල static chat UI එකක් (`index.html`). සිංහල / English chatbot එකක් browser එකෙන්ම
පාවිච්චි කරන්න. InfinityFree (free hosting) වලින් host කරන්න පුළුවන්.

## 1. InfinityFree එකේ setup කරන්න
1. https://infinityfree.net වෙත sign up කරලා hosting account එකක් හදන්න.
2. Subdomain එකක් ගන්න (උදා: `mybot.epizy.com`).
3. **File Manager** (හෝ FTP) මගින් `index.html` එක `htdocs/` folder එකට upload කරන්න.
4. Browser එකේ subdomain එකට යන්න → chat UI එක පෙනවා.

## 2. UI එක settings කරන්න (⚙️ button)
UI එකේ settings වලට යන්න. දෙකකින් එකක් තෝරන්න:

### A) FREE API (Groq) — 70B+, keys ඔයාගේම
- **API Base URL**: `https://api.groq.com/openai/v1`
- **API Key**: `gsk_xxx` (console.groq.com වෙතින්)
- **Model**: `llama-3.3-70b-versatile`
- ⚠️ Key එක browser එකේම (localStorage) තියෙනවා — ඔයාගේම පුද්ගලික පිටුවට පමණයි.

### B) Self-hosted api.py (open-source, no key)
- **API Base URL**: `http://YOUR_PUBLIC_HOST:8000/v1`
- **API Key**: හිහට් තියන්න
- **Model**: `local` (හෝ ollama/vllm model name)
- api.py එක CORS `*` ලබාදෙන නිසා cross-origin call එක වැඩ කරනවා.
- ⚠️ api.py server එක අන් තැනක (GPU server) public කරලා තියෙන්න ඕනේ.

## 3. Notes
- InfinityFree static files පමණයි serve කරනවා (Python/GPU backend නැහැ).
- ඒ නිසා model/API එක වෙනත් තැනක් වලයි (Groq free / ඔයාගේ server).
- CORS එක api.py එකේ allow කලා; free API (Groq/OpenRouter) ද CORS අනුමත කරනවා.
