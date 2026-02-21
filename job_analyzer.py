import streamlit as st
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import instructor
from groq import Groq
from dotenv import load_dotenv

# ==============================================================================
# 1. SETUP & SECURITATE
# ==============================================================================
st.set_page_config(page_title="GenAI Headhunter", page_icon="🕵️", layout="wide")

# Încărcăm variabilele din fișierul .env
load_dotenv()

# Încercăm să luăm cheia din OS (local) sau din Streamlit Secrets (cloud)
api_key = os.getenv("GROQ_API_KEY")

# Fallback pentru Streamlit Cloud deployment
if not api_key and "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

# Validare critică: Dacă nu avem cheie, oprim aplicația aici.
if not api_key:
    st.error("⛔ EROARE CRITICĂ: Lipsește `GROQ_API_KEY`.")
    st.info("Te rog creează un fișier `.env` în folderul proiectului și adaugă: GROQ_API_KEY=cheia_ta_aici")
    st.stop()

# Configurare Client Groq Global (pentru a nu-l reinițializa constant)
client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.TOOLS)

# Sidebar Informativ (Fără input de date sensibile)
with st.sidebar:
    st.header("🕵️ GenAI Headhunter")
    st.success("✅ API Key încărcat securizat")
    st.markdown("---")
    st.write("Acest tool demonstrează:")
    st.write("• Web Scraping (BS4)")
    st.write("• Secure Env Variables")
    st.write("• Structured Data (Pydantic)")


# ==============================================================================
# 2. DATA MODELS (PYDANTIC SCHEMAS)
# ==============================================================================
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class SalaryRange(BaseModel):
    min: int = Field(..., ge=0, description="Salariul minim")
    max: int = Field(..., ge=0, description="Salariul maxim")
    currency: str = Field(..., description="Moneda (EUR, USD, RON etc.)")

    @model_validator(mode="after")
    def check_range(self):
        if self.max < self.min:
            raise ValueError("Salary max nu poate fi mai mic decât min.")
        return self


class Location(BaseModel):
    city: str = Field(..., description="Orașul jobului")
    country: str = Field(..., description="Țara jobului")
    is_remote: bool = Field(False, description="True dacă jobul este remote sau hibrid")


class RawExtraction(BaseModel):
    role_title: str | None = None
    company_name: str | None = None
    seniority: Literal["Intern", "Junior", "Mid", "Senior", "Lead", "Architect"] | None = None
    tech_stack: List[str]
    salary_range: Optional[SalaryRange] = None  
    location: Optional[Location] = None
    requirements: List[str]
    
class RedFlag(BaseModel):
    severity: Literal["low", "medium", "high"] = Field(
        ..., description="Nivel gravitate"
    )
    category: Literal["toxicity", "vague", "unrealistic"] = Field(
        ..., description="Categoria problemei"
    )
    description: str = Field(..., description="Descriere scurtă")

    
class JobAnalysis(BaseModel):
    summary: str
    match_score: int = Field(..., ge=0, le=100)
    red_flags: List[RedFlag] = Field(default_factory=list)
    negotiation_tip: str
    interview_questions: List[str]

# ==============================================================================
# 3. UTILS - SCRAPER (Colectare Date)
# ==============================================================================

def scrape_clean_job_text(url: str, max_chars: int = 3000) -> str:
    """
    Descarcă pagina și returnează un text curat, optimizat pentru contextul LLM.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Error: Status code {response.status_code}"
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Eliminăm elementele inutile care consumă tokeni
        for junk in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            junk.decompose()
            
        # Extragem textul și eliminăm spațiile multiple
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        
        return text[:max_chars] 
        
    except Exception as e:
        return f"Scraping Error: {str(e)}"

# ==============================================================================
# 4. AI SERVICE LAYER - EXTRACTION (Logica LLM)
# ==============================================================================

def extract_job_details_with_ai(text: str) -> RawExtraction:
    """
    Trimite textul curățat către Groq și returnează obiectul structurat.
    """
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=RawExtraction,
        messages=[
            {
                "role": "system", 
                "content": (
                    "Extrage informatii din textul jobului cu obiectivitate. "
                    "Răspunde strict în formatul cerut."
                )
            },
            {
                "role": "user", 
                "content": f"Analizează acest job description:\n\n{text}"
            }
        ],
        temperature=0.0,
    )

# ==============================================================================
# 5. AI SERVICE LAYER - INTERPRETATION (Logica LLM)
# ==============================================================================

def interpret_job_details_with_ai(input: RawExtraction) -> JobAnalysis:
    """
    Trimite datele extrase către Groq și returnează interpretarea intr-un obiect structurat.
    """
    return client.chat.completions.create(
        model="qwen/qwen3-32b",
        response_model=JobAnalysis,
        messages=[
            {
                "role": "system", 
                "content": (
                    "Ești un Recruiter Expert în IT. Analizează jobul cu obiectivitate. "
                    "Daca location.is_remote e True dar requirements sugereaza prezenta la birou, adauga un red flag cu severity medium si category unrealistic."
                    "Răspunde strict în formatul cerut."
                )
            },
            {
                "role": "user", 
                "content": f"Analizează acest job:\n\n{input}"
            }
        ],
        temperature=0.7,
    )

# ==============================================================================
# 5. UI - APLICAȚIA STREAMLIT
# ==============================================================================

st.title("🕵️ GenAI Headhunter Assistant")
st.markdown("Transformă orice Job Description într-o analiză structurată folosind AI.")

# Tab-uri
tab1, tab2 = st.tabs(["🚀 Analiză Job", "📊 Market Scan (Batch)"])

# --- TAB 1: ANALIZA UNUI SINGUR LINK ---
with tab1:
    st.subheader("Analizează un Job URL")
    url_input = st.text_input("Introdu URL-ul:", placeholder="https://...")
    
    if st.button("Analizează Job", key="btn_single"):
        if not url_input:
            st.warning("Te rugăm introdu un URL.")
        else:
            with st.spinner("🕷️ Scraping & 🤖 AI Analysis..."):
                raw_text = scrape_clean_job_text(url_input)
            
            if "Error" in raw_text:
                st.error(raw_text)
            else:
                try:
                    raw_data = extract_job_details_with_ai(raw_text)
                    st.info(raw_data)
                    
                    interpreted_data = interpret_job_details_with_ai(raw_data)
                    st.info(interpreted_data)
                    
                    # -- DISPLAY --
                    st.divider()

                    # ===============================
                    # HEADER SECTION
                    # ===============================
                    col_h1, col_h2 = st.columns([3, 1])

                    with col_h1:
                        st.markdown(f"## {raw_data.role_title}")
                        st.caption(f"🏢 {raw_data.company_name} • 🎯 {raw_data.seniority}")

                    with col_h2:
                        score_color = "normal" if interpreted_data.match_score >= 75 else "inverse"
                        st.metric("Quality Score", f"{interpreted_data.match_score}/100", delta_color=score_color)

                    # ===============================
                    # QUICK STATS ROW
                    # ===============================
                    city = raw_data.location.city if raw_data.location else "Nespecificat"
                    country = raw_data.location.country if raw_data.location else ""
                    is_remote = raw_data.location.is_remote if raw_data.location else False

                    salary_text = "Nespecificat"
                    if raw_data.salary_range:
                        salary_text = f"{raw_data.salary_range.min}-{raw_data.salary_range.max} {raw_data.salary_range.currency}"

                    c1, c2, c3, c4 = st.columns(4)

                    c1.info(f"📍 **Locație:** {city} {country}")
                    c2.info(f"🏠 **Remote:** {'Da' if is_remote else 'Nu'}")
                    c3.success(f"🛠️ **Tehnologii:** {len(raw_data.tech_stack)}")
                    c4.warning(f"🚩 **Red Flags:** {len(interpreted_data.red_flags)}")

                    # ===============================
                    # SALARY SECTION
                    # ===============================
                    st.markdown("### 💰 Salary")
                    if raw_data.salary_range:
                        st.success(f"Interval salarial: **{salary_text}**")
                    else:
                        st.caption("Nu este specificat un interval salarial.")

                    # ===============================
                    # SUMMARY SECTION
                    # ===============================
                    st.markdown("### 📝 Rezumat")
                    st.info(interpreted_data.summary)

                    # ===============================
                    # TECH STACK SECTION
                    # ===============================
                    st.markdown("### 🛠️ Tech Stack")

                    if raw_data.tech_stack:
                        tech_badges = " ".join([f"`{tech}`" for tech in raw_data.tech_stack])
                        st.markdown(tech_badges)
                    else:
                        st.caption("Nu au fost identificate tehnologii clare.")

                    # ===============================
                    # RED FLAGS SECTION
                    # ===============================
                    st.markdown("### 🚩 Avertismente")

                    if interpreted_data.red_flags:
                        for flag in interpreted_data.red_flags:
                            label = f"[{flag.severity.upper()} | {flag.category.upper()}]"
                            
                            if flag.severity == "high":
                                st.error(f"{label} {flag.description}")
                            elif flag.severity == "medium":
                                st.warning(f"{label} {flag.description}")
                            else:
                                st.info(f"{label} {flag.description}")
                    else:
                        st.success("Nu au fost detectate red flags majore.")

                except Exception as e:
                    st.error(f"Eroare AI: {str(e)}")


# --- TAB 2: BATCH PROCESSING ---
with tab2:
    st.subheader("📊 Compară mai multe joburi")
    urls_text = st.text_area("Paste URL-uri (unul pe linie):", height=150)
    
    if st.button("Scanează Piața", key="btn_batch"):
        urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
        
        if not urls:
            st.warning("Nu ai introdus link-uri.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, link in enumerate(urls):
                status_text.text(f"Analizez {i+1}/{len(urls)}...")
                text = scrape_clean_job_text(link)
                
                if "Error" not in text:
                    try:
                        raw_data = extract_job_details_with_ai(text)
                    
                        interpreted_data = interpret_job_details_with_ai(raw_data)
                        
                        results.append({
                            "Role": raw_data.role_title,
                            "Company": raw_data.company_name,
                            "Seniority": raw_data.seniority,
                            "Tech": raw_data.tech_stack,
                            "Score": interpreted_data.match_score
                        })
                    except:
                        pass # Continuăm chiar dacă unul crapă
                
                progress_bar.progress((i + 1) / len(urls))
            
            status_text.text("Gata!")
            
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Grafic simplu
                st.bar_chart(df['Seniority'].value_counts())
