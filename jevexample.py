from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
import os
from dotenv import load_dotenv
load_dotenv()
client = TypeSafeClient(api_key=os.getenv("JEV_API_KEY"))

#ticket = "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP."
ticket = "Payment   failed in striped getting 502 error, losing sales  because of 502 error . requires immediate attentation"
response = client.system_one(
state=ticket,
questions={
"department": Choice(
instructions="Which team should handle this",
criteria={
"billing": "Payment or subscription issues",
"technical": "Bugs or integration problems",
"sales": "Pricing or account questions",
},
),
"frustration": Score(
instructions="How frustrated the customer appears",
criteria=[
"Calm, just stating facts",
"Frustrated but civil",
"Very angry, strong language",
],
),
"is_urgent": Noul(
instructions="The message conveys urgency or time-sensitivity",
),

},
)

print(response.answers["department"].choice) # "technical"
print(response.answers["frustration"].score) # 1.0
print(response.answers["is_urgent"].noul) # 1.0

