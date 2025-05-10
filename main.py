from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="All-in-ONE GPT Clinical API")

class ClinicalRequest(BaseModel):
    task: str
    patient_data: str

@app.post("/clinical/")
async def clinical_tool(request: ClinicalRequest):
    try:
        system_prompt = """
        You are 'All in ONE', a clinical GPT used in inpatient hospital settings.
        You interpret clinical notes, labs, imaging reports, and provide:
        - H&P exam notes
        - DVT risk/prophylaxis decisions
        - InterQual level-of-care assessments
        - Consult text templates
        - Clinical issue flagging
        - Handoff summaries
        Return responses in structured, professional, HIPAA-compliant format.
        """

        user_input = f"Task: {request.task}\n\nData:\n{request.patient_data}"

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        return {"result": response['choices'][0]['message']['content']}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# This allows the app to run locally and on Railway using the correct port
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
