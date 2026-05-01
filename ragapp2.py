import sys, os, json, math
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
# ── OpenRouter client (Qwen) ──────────────────────────────────
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
)
QWEN_MODEL   = "openrouter/free"
 
# ════════════════════════════════════════════════════════════
# CHANGE THESE FOR EACH BUSINESS CLIENT
# ════════════════════════════════════════════════════════════
BOT_NAME      = "Caterbot"
BUSINESS_NAME = "Desi Cuisine"
THEME_COLOR   = "#0ea5e9"
WELCOME_MSG   = "Hi! 👋 I'm CaterBot. Ask me about our services, suggestions for budget friendly items, quantity of dishes required, or fees!"
VECTOR_STORE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_store.json")
 
# ════════════════════════════════════════════════════════════
# PROMPT 1 — SYSTEM PROMPT
# ════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
You are a helpful, friendly, and professional customer service assistant for a catering business. Your job is to assist customers with:
1. Checking prices of dishes
2. Calculating how much quantity of a dish or multiple dishes to order for a given number of people
3. Suggesting a complete menu within a given budget
4. Answering any questions about available dishes and drinks
PERSONALITY:
- Always be warm, polite, and confident.
- Speak clearly and simply.
- Always respond in the same language the customer uses (Urdu or English).
- If a customer writes in Urdu, reply in Urdu. If in English, reply in English.
STRICT RULES:
- Only answer based on the information in your knowledge base.
- Never guess, invent, or assume any price, dish, or quantity not provided to you.
- If a customer asks about a dish not in your knowledge base, say:
  "I'm sorry, that item is not currently on our menu."
- Always ask after dishes whether they want carbonated drinks or cardamom tea.  
- Never give a food-only total. Always include drinks in every event estimate.
- Always mention bottle sizes when quoting drinks:
  carbonated drinks are 250 ml per bottle, mineral water is 1.5 litre per bottle.
- Always mention the water bottle return/refund policy when quoting water.
- If customer wants you to suggest and dont tell about month or season of event ask them and in winter season prefer them cardamom tea over carbonated drink and Gajar ka Halva in dessert.
- When a customer asks about multiple dishes, always ask which dish is their main dish before calculating.
- Always round UP quantities. Never round down.
- Always end with: "Is there anything else I can help you with?"
GOAL:
Help the customer make the best decision for their event confidently and accurately.
"""
 
# ════════════════════════════════════════════════════════════
# PROMPT 2 — LOGIC PROMPT
# ════════════════════════════════════════════════════════════
LOGIC_PROMPT = """
RAG ANSWER LOGIC RULES — CATERING BOT
======================================
--- RULE 1: SINGLE DISH QUANTITY ---
When customer asks: "How much [dish] do I need for N people?"
 
Step 1: Identify dish type.
  Gravy/Rice dish: kg needed = ceil(N ÷ 12) × 5
  Dry/BBQ dish:    kg needed = ceil(N ÷ 7) × 5
 
Step 2: Cost = kg needed × price per kg
 
Step 3: Present clearly.
  Example: "You need 10 kg of Chicken Biryani for 50 people,
  which will cost Rs. 11,000."
 
--- RULE 2: MULTI-DISH QUANTITY ---
CORE PRINCIPLE:
Total food across ALL dishes combined = enough for N people.
Never calculate each dish for N people independently.
 
STEP 1 — ALWAYS ASK THE CUSTOMER FIRST:
"Which dish would you like to be the main dish of your menu?"
 
STEP 2 — CALCULATE TOTAL KG NEEDED FOR N PEOPLE:
  Total kg = ceil(N ÷ 12) × 5
 
STEP 3 — SPLIT TOTAL KG BY NUMBER OF DISHES:
  2 dishes: Main 60%+1kg, Other 40%
  3 dishes: Main 50%+1kg, Other1 25%, Other2 25%
  4 dishes: Main 40%+1kg, Other1 25%, Other2 20%, Other3 15%
  Round each UP to nearest whole kg.
 
STEP 4 — CALCULATE COST PER DISH:
  Cost = kg of that dish × its price per kg
 
STEP 5 — PRESENT WITH FULL BREAKDOWN.
 
--- RULE 3: BUDGET-BASED MENU ---
STEP 1 — ASK THESE QUESTIONS FIRST (if not already provided):
  a) How many people?
  b) Type of event?
  c) Dish preferences or items to avoid?
  d) Should dessert be included?
  e) Chicken, beef, or mutton preference?
  f) Drinks preference?
 
STEP 2 — RESERVE DRINKS BUDGET FIRST (compulsory):
  Carbonated drinks: ceil(N × 1.20) × Rs. 60
  Tea: ceil(N÷4) × Rs. 600
  Mineral water: ceil(N ÷ 2) × Rs. 110
  Remaining food budget = total budget − drinks cost
 
STEP 3 — BUILD FOOD MENU WITHIN REMAINING BUDGET.
STEP 4 — CHECK TOTAL VS BUDGET and fix if over.
STEP 5 — PRESENT FULL MENU BREAKDOWN.
 
--- RULE 4: DRINKS (ALWAYS COMPULSORY) ---
Always ask preference: tea or carbonated drink.
Cardamom Tea: Rs. 600 per litre (cups included), 1 litre serves 4 people.
Carbonated: 250ml bottles, Rs. 60/bottle, ceil(N × 1.20) bottles.
Mineral water: 1.5L bottles, Rs. 110/bottle, ceil(N ÷ 2) bottles.
Return policy: unused sealed water bottles refunded at Rs. 110/bottle.
 
--- RULE 5: GENERAL RULES ---
- Always round UP. Never round down.
- Never invent prices, dishes, or quantities.
- Never give food-only total without drinks.
- Respond in the same language as the customer.
"""
 
# ════════════════════════════════════════════════════════════
# RAG FUNCTIONS
# ════════════════════════════════════════════════════════════
 
def cosine_similarity(a, b):
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
 
TFIDF_META   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfidf_meta.json")
 
@st.cache_resource
def load_store():
    with open(VECTOR_STORE, "r", encoding="utf-8") as f:
        return json.load(f)
 
@st.cache_resource
def load_vectorizer():
    with open(TFIDF_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    vectorizer = TfidfVectorizer(max_features=512, vocabulary=meta["vocabulary"])
    vectorizer.idf_ = meta["idf"]
    import numpy as np
    vectorizer.idf_ = np.array(meta["idf"])
    return vectorizer
 
def retrieve(question, top_k=7):
    store      = load_store()
    vectorizer = load_vectorizer()
    q_vec      = vectorizer.transform([question]).toarray()[0].tolist()
 
    scored = [
        {"text": c["text"], "source": c["source"],
         "score": cosine_similarity(q_vec, c["embedding"])}
        for c in store
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
 
def generate(question, chunks, history):
    context = ""
    for c in chunks:
        context += f"\n[Source: {c['source']}]\n{c['text']}\n"
        context += "-" * 30
 
    full_system = f"""{SYSTEM_PROMPT}
 
{LOGIC_PROMPT}
 
CONTEXT FROM DOCUMENTS (answer ONLY from this):
{context}"""
 
    messages = [{"role": "system", "content": full_system}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
 
    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content
 
def rag_answer(question, history=[]):
    chunks  = retrieve(question)
    answer  = generate(question, chunks, history)
    sources = list(set(c["source"] for c in chunks))
    return answer, sources
 
# ════════════════════════════════════════════════════════════
# ACCURACY TESTS
# ════════════════════════════════════════════════════════════
TEST_CASES = [
    {"question": "What is the price of 8kg Biryani?",                      "expected": "Rs.8800"},
    {"question": "How much quantity of gulab jamun required for 7 people?", "expected": "1 kg"},
]
 
def run_accuracy_test():
    print("\n" + "=" * 55)
    print("  ACCURACY TEST — RAG CHATBOT")
    print("=" * 55)
 
    if not os.path.exists(VECTOR_STORE):
        print("❌ vector_store.json not found. Run rag_ingest.py first.")
        return
 
    passed, failed = 0, 0
 
    for i, test in enumerate(TEST_CASES):
        question, expected = test["question"], test["expected"]
        try:
            reply, sources = rag_answer(question)
            ok = expected.lower() in reply.lower()
            passed += 1 if ok else 0
            failed += 0 if ok else 1
            status = "✅ PASS" if ok else "❌ FAIL"
            print(f"\n[{i+1}] {status}")
            print(f"     Q: {question}")
            print(f"     Expected: '{expected}'")
            print(f"     Got: {reply[:120]}...")
            print(f"     Sources: {sources}")
        except Exception as e:
            failed += 1
            print(f"\n[{i+1}] ❌ ERROR — {question}\n     {str(e)}")
 
    total    = passed + failed
    accuracy = round((passed / total) * 100) if total > 0 else 0
    print("\n" + "=" * 55)
    print(f"  RESULT: {passed}/{total} passed — Accuracy: {accuracy}%")
    print("  ✅ GOOD" if accuracy >= 80 else "  ⚠️ OK" if accuracy >= 60 else "  ❌ LOW")
    print("=" * 55 + "\n")
# ════════════════════════════════════════════════════════════
# FLASK ROUTES (same as your original code)
# ══════════════"""══════════════════════════════════════════════
""" 
@app.route("/")
def home():
    return render_template(
        "index.html",
        bot_name      = BOT_NAME,
        business_name = BUSINESS_NAME,
        theme_color   = THEME_COLOR,
        welcome_msg   = WELCOME_MSG
    )
 
 
@app.route("/chat", methods=["POST"])
def chat():
    data     = request.json
    question = data.get("message", "")
    history  = data.get("history", [])
 
    if not question:
        return jsonify({"error": "No message"}), 400
 
    if not os.path.exists(VECTOR_STORE):
        return jsonify({
            "reply": "Knowledge base not ready. Run ingest.py first."
        })
 
    answer, sources = rag_answer(question, history)
    return jsonify({"reply": answer, "sources": sources})
 
# ════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_accuracy_test()
    else:
        print(f"  {BOT_NAME} for {BUSINESS_NAME} — starting...")
        print(f"  Open: http://localhost:5000")
        print(f"  To test accuracy: python app.py test")
        app.run(debug=True, port=5000)"""
