from fastapi import FastAPI
from fastapi.responses import Response

from telemetry import setup_telemetry
import logging
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import aiofiles

requests_total = Counter('requests_total', 'Total number of requests')

app = FastAPI()
tracer = setup_telemetry(app, service_name="demo-fastapi-otel")   # <-- this line must exist
logger = logging.getLogger(__name__)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/home")
def home():
    requests_total.inc()
    return "OK"

@app.get("/")
async def root():
    return {"message": "hello world"}

@app.get("/read")
async def read_requirements():
    with tracer.start_as_current_span("read-requirements") as span:
        try:
            span.set_attribute("file.name", "requirements.txt")
            with open("requirements2txt", "r") as f:
                content = f.read()
            span.add_event("requirements_read_success", {"file_size": len(content)})
            logger.info("Successfully read requirements.txt")
            return {"content": content}
        except FileNotFoundError as e:
            span.set_attribute("error", True)
            span.add_event("file_not_found", {"error": str(e)})
            logger.exception("requirements.txt not found: {e}")
            return {"error": "requirements.txt file not found"}
        except Exception as e:
            span.set_attribute("error", True)
            span.add_event("read_error", {"error": str(e)})
            logger.exception(f"Error reading requirements.txt: {e}")
            return {"error": str(e)}

@app.get("/work")
async def do_work():
    with tracer.start_as_current_span("process-work") as span:   # uses module-level `tracer`
        span.set_attribute("work.type", "example")
        logger.info("Doing work now")
        result = 1 + 1
        span.add_event("computation done", {"result": result})
        return {"result": result}