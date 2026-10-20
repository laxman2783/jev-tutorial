import logging
import secrets

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

resource = Resource.create({"service.name": "dice-server"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

LoggingInstrumentor().instrument(set_logging_format=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - trace_id=%(otelTraceID)s span_id=%(otelSpanID)s - %(message)s",
    handlers=[
        logging.FileHandler("app3.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,
)

logger = logging.getLogger()
app = FastAPI()
FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)


@app.get("/hello")
def hello():
    with trace.get_tracer(__name__).start_as_current_span("hello_handler"):
        number = secrets.randbelow(100) + 1
        message = f"hello world - random number: {number}"
        logger.info(message)
        return {"message": message}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("f1:app", host="0.0.0.0", port=8086, reload=False)
