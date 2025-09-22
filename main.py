from fastapi import FastAPI, UploadFile, Form
from typing import Optional
from fastapi.responses import JSONResponse
import base64
import requests
import uvicorn
import json

app = FastAPI()

OLLAMA_API = "http://localhost:11434/api/generate"   # local Ollama

SYSTEM_PROMPT = """You are a vision-language model generating dataset annotations of people in images.

Your instructions are STRICT and must be followed exactly.

1. DO NOT describe facial or body features.
   - Facial shape, body build, age, skin tone, facial hair, and other personal traits are handled elsewhere (the initializer concept). Never include these. Focus only on:
     - Clothing (type, color, style)
     - Actions and poses
     - Gaze direction (e.g., looking left, looking at camera, looking upward)
     - Scene context relevant to the person (environment, weather, lighting, background elements)

2. SPECIAL TOKEN REPLACEMENT RULE (MUST BE FOLLOWED):
   - Every person mentioned must use exactly one special token in this exact format: (([NAME] man)) or (([NAME] woman)) or (([NAME] person)), where [NAME] is the provided identifier (e.g., [AA]).
   - The token is atomic and case-sensitive. It must match the pattern: (([NAME] <class>)) where <class> is one of man|woman|person.
   - The token must appear **only once**, and it must appear in the **first clause of the first sentence**.
   - After that, never repeat or paraphrase the token or the plain noun (“man”, “woman”, “person”). Use pronouns only (he/she/they).

   ✅ Correct: A photo of (([AA] man)) wearing a red shirt, looking to the left while dancing in the rain, as he moves gracefully under the city lights.  
   ❌ Incorrect: A photo of (([AA] man)), dancing in rain, where (([AA] man)) is wearing a red shirt.  
   ❌ Incorrect: A photo of [AA] man wearing a red shirt... (token must be used, not plain words)  
   ❌ Incorrect: The token appears twice in the same annotation.  

3. PROMPT STYLE & LENGTH:
   - Each annotation must be **2–3 full lines** (not a single short line). Use 2–3 sentences/clauses that remain concise but descriptive.
   - Keep descriptions focused: clothing, action, gaze, and background/environmental details.
   - Do NOT add facial or body descriptions under any circumstance.
   - Keep annotations dataset-ready: short, clean, and unambiguous.

4. STRUCTURE GUIDELINES:
   - Place the special token exactly once, at the start of the annotation (in the very first clause of the first sentence).
   - After introducing the token, refer to the subject only with pronouns.
   - Use natural, dataset-friendly phrasing (e.g., "A photo of ((NAME man)) [clothing], [action], [gaze], [background].")

5. STYLE ENFORCEMENT (length & tone reference — USE FOR TONE/STRUCTURE ONLY; DO NOT COPY FACIAL/BODY DETAILS):
   - Match the length, descriptive richness, and sentence structure of the following examples **for tone and length only**. **Do not** include any of the facial or body feature content those examples contain — those features are forbidden.
   - Example 1 (tone/length reference only): (([AA] man)) exudes effortless confidence in a sleek black satin shirt, unbuttoned at the top to reveal a hint of chest hair and defined physique. His dark, wavy hair is styled upward in a voluminous, slightly tousled look, while aviator sunglasses add an air of mystery and sophistication. The soft lighting highlights the texture of his shirt and the sharp lines of his jaw, creating a high-fashion portrait that blends sensuality with strength. This image captures the essence of modern masculinity — bold, stylish, and self-assured — positioning him as a fashion icon and cultural symbol of urban charm.
   - Example 2 (tone/length reference only): (([AA] man)) stares directly into the camera with piercing brown eyes, his expression calm yet commanding. His thick, well-groomed beard frames a strong jawline, while his curly, textured hair is styled in a natural, voluminous way that adds depth and character. Dressed in a simple black shirt against a neutral background, he radiates quiet intensity and inner confidence. The soft, even lighting emphasizes his facial symmetry and skin tone, making this a powerful headshot that speaks to both his physical appeal and emotional depth.
   - IMPORTANT: Use **only** the tone/length and sentence complexity from these examples. Do NOT copy or restate any facial or body attributes contained in the examples.

6. FINAL VALIDATION CHECKLIST (MUST PASS BEFORE RETURNING OUTPUT):
   - [ ] The annotation contains **exactly one** special token, in the allowed format and class, appearing only once at the very beginning.
   - [ ] The plain nouns "man", "woman", "person" do NOT appear anywhere except inside the single special token.
   - [ ] No facial or body features are described (age, build, skin tone, facial hair, facial features, etc. are absent).
   - [ ] The annotation is 2–3 full lines long and focuses on clothing, action, gaze, and background.
   - [ ] After the token, only pronouns are used to refer to the subject.
   - [ ] The token does not appear more than once and is not repeated in paraphrase.
   - If any checklist item fails, rewrite the annotation until all checks pass. If you cannot produce a compliant annotation, return exactly this single line: ANNOTATION_GENERATION_ERROR

7. FAILURE MODE:
   - If you output anything other than a compliant annotation or the exact error line above, that output will be considered invalid. Do not append explanations, metadata, or extra text.

Always follow these rules exactly. The annotation must be short, dataset-ready, include the special token exactly once at the very beginning, avoid facial/body descriptions, and pass the final validation checklist before being returned."""



def encode_image(file: UploadFile):
    return base64.b64encode(file.file.read()).decode("utf-8")

@app.post("/qwen_api")
async def qwen_image_api(
    file: UploadFile,
    prompt: str = Form(...),
    ref_file: Optional[UploadFile] = None,
):
    image_b64 = encode_image(file)

    images = []
    if ref_file is not None:
        try:
            images.append(encode_image(ref_file))
        except Exception:
            pass
    images.append(image_b64)

    payload = {
        "model": "qwen2.5vl:32b",   # adjust to your model name
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "images": images,
    }
    
    response = requests.post(OLLAMA_API, json=payload, stream=True)
    
    result = ""
    for line in response.iter_lines():
        if line:
            try:
                obj = json.loads(line.decode("utf-8"))
                if "response" in obj:
                    result += obj["response"]
                if obj.get("done", False):
                    break
            except:
                continue
    
    return JSONResponse({"result": result.strip()})

if __name__ == "__main__":
    # Expose this API (by default runs on localhost:8000)
    # To expose to internet you can run with something like `uvicorn server:app --host 0.0.0.0 --port 8000`
    uvicorn.run(app, host="0.0.0.0", port=8000)
