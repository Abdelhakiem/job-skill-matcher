from fastapi import  FastAPI, Body

import predictor

predictor.initialize_model()
app = FastAPI()

@app.post("/inference/suggest_jobs")
def suggest_jobs(body = Body()):
    """Suggest jobs based on the provided skills."""
    skills = body.get("skills", [])
    if not skills:
        return {"error": "No skills provided."}
    
    return predictor.predict_job_probabilities(skills)



# uvicorn server:app --port 8001 --host 0.0.0.0 
# http://localhost:8001/docs