
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from diabetes.constants import APP_HOST, APP_PORT
from diabetes.pipline.prediction_pipeline import diabetesData, diabetesClassifier
from diabetes.pipline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Pregnancies: Optional[str] = None
        self.Glucose: Optional[str] = None
        self.BloodPressure: Optional[str] = None
        self.SkinThickness: Optional[str] = None
        self.Insulin: Optional[str] = None
        self.BMI: Optional[str] = None
        self.DiabetesPedigreeFunction: Optional[str] = None
        self.Age: Optional[str] = None
        
        

    async def get_diabetes_data(self):
        form = await self.request.form()
        self.continent = form.get("Pregnancies")
        self.education_of_employee = form.get("Glucose")
        self.has_job_experience = form.get("BloodPressure")
        self.requires_job_training = form.get("SkinThickness")
        self.no_of_employees = form.get("Insulin")
        self.company_age = form.get("BMI")
        self.region_of_employment = form.get("DiabetesPedigreeFunction")
        self.prevailing_wage = form.get("Age")
        

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "diabetes.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_diabetes_data()
        
        diabetes_data = diabetesData(
                                Pregnancies= form.Pregnancies,
                                Glucose=  form.Glucose,
                                Insulin=  form.Insulin,
                                BloodPressure= form.BloodPressure,
                                SkinThickness= form.SkinThickness,
                                BMI = form.BMI,
                                DiabetesPedigreeFunction= form.DiabetesPedigreeFunction,
                               
                                Age= form.Age,
                                
                                )
        
        diabetes_df = diabetesData.get_diabetes_input_data_frame()

        model_predictor = diabetesClassifier()

        value = model_predictor.predict(dataframe=diabetes_df)[0]

        status = None
        if value == 1:
            status = "you have diabetes"
        else:
            status = "you dont have diabetes"

        return templates.TemplateResponse(
            "diabetes.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)