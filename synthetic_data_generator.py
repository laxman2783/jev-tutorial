"""
Synthetic conversation generator for the Conversational Insights script.
Creates realistic support conversations and loads them into BigQuery.

Run this FIRST, then run conversational_insights.py unchanged.

    pip install google-cloud-bigquery pandas
"""
import random
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import bigquery

PROJECT = "project-819150c4-7f48-40eb-b26"   # same values as the insights script
DATASET = "analytics"
DEST_TABLE = "conversations"
N = 400
random.seed(42)

TOPICS = [
 dict(name="billing", openers=[
  "I was charged twice on my credit card for order #84210 and I need one of the charges reversed.",
  "My invoice this month is $89 but my plan is supposed to be $59. Why was I overcharged?",
  "I requested a refund two weeks ago and the money still hasn't shown up in my account.",
  "You billed me for a full year even though I cancelled in March. I want a prorated refund.",
  "There's a $12.99 charge I don't recognize on my statement from last Tuesday."],
  agent=[
  "I'm sorry about that. Let me pull up your billing history and see what happened.",
  "I can see the duplicate charge on my end. I'll issue a reversal right now — it should post in 3 to 5 business days.",
  "Thanks for flagging this. I'm escalating your refund to the billing team for review."],
  followups=[
  "This is the second time this has happened. I'm really frustrated.",
  "Okay, but can you confirm I'll get an email once the refund goes through?",
  "I need this resolved today, I can't keep calling about the same charge."],
  closings=[
  "Your refund of $89.00 has been processed. You should see it within 5 business days.",
  "I've corrected the invoice and applied a credit to your account."]),

 dict(name="tech_support", openers=[
  "The mobile app crashes every time I try to upload a photo. I'm on the latest version.",
  "Since yesterday's update I can't sync my data — it just spins forever and then times out.",
  "The dashboard shows an error 500 whenever I export a report to PDF.",
  "My notifications stopped working completely after I changed my phone.",
  "The search function returns no results even for items I know exist."],
  agent=[
  "Thanks for reporting this. Can you tell me your app version and device model so I can reproduce it?",
  "That sounds like a known issue with the latest release. Our engineers are working on a fix.",
  "Let's try clearing the app cache first — that resolves the sync problem for most users."],
  followups=[
  "I already reinstalled the app twice and it's still broken.",
  "It's version 4.2.1 on Android 14, if that helps.",
  "How long until the fix is released? This is blocking my work."],
  closings=[
  "The fix is included in version 4.2.2, rolling out tomorrow. Thanks for your patience.",
  "I've logged this as ticket #55231 and you'll get an update by email."]),

 dict(name="shipping", openers=[
  "My order was supposed to arrive last Friday and tracking hasn't updated in six days.",
  "The courier marked my package as delivered but there's nothing at my door.",
  "I paid for express two-day shipping and it's been a week. I want the shipping fee refunded.",
  "Half of my order arrived but two items are missing from the box."],
  agent=[
  "I understand how frustrating that is. Let me trace the shipment with the carrier.",
  "I can see the package is held at the regional depot. I'll arrange redelivery for tomorrow.",
  "I'll refund the express shipping fee right away and open an investigation on the missing items."],
  followups=[
  "This was a birthday gift and now it's going to be late.",
  "Can you just send a replacement instead of waiting for the carrier?",
  "The tracking number is 1Z8842W903441."],
  closings=[
  "A replacement is on its way with priority shipping at no extra cost.",
  "The carrier confirmed delivery to a neighbor at number 14 — please check with them."]),

 dict(name="account_access", openers=[
  "I can't log into my account — it says my password is incorrect but I just reset it.",
  "I'm not receiving the two-factor authentication code on my phone.",
  "How do I change the email address on my account? The settings page won't save.",
  "My account got locked after too many login attempts. How do I unlock it?"],
  agent=[
  "I can help with that. I'll send a fresh reset link to your registered email.",
  "Let's verify your identity first, then I can disable the old 2FA and set up a new one.",
  "The lock clears automatically after 30 minutes, but I can unlock it manually right now."],
  followups=[
  "I got the reset link, but it says it's expired already.",
  "My phone number changed, so I can't receive codes at the old one.",
  "Okay, I'm back in now. Thanks."],
  closings=[
  "Your account is unlocked and 2FA is reconfigured. You're all set.",
  "Email updated successfully. You'll get a confirmation shortly."]),

 dict(name="praise", openers=[
  "Just wanted to say your support team was amazing — resolved my issue in ten minutes.",
  "I've been a customer for three years and the service keeps getting better. Thank you!",
  "The new dashboard is fantastic, so much faster than before. Great work.",
  "Shoutout to Maria on your team, she went above and beyond to help me yesterday."],
  agent=[
  "Thank you so much for the kind words! I'll pass your feedback to the team.",
  "We're thrilled to hear that. Thanks for being a loyal customer!"],
  followups=[
  "Keep up the good work, you've earned a customer for life.",
  "Already recommended you to two friends."],
  closings=["Thanks again for taking the time to share this!"]),

 dict(name="feature_request", openers=[
  "It would be great if you added dark mode to the mobile app.",
  "Do you plan to support exporting reports to Excel? PDF only is limiting.",
  "I'd love a way to bulk-edit my saved items instead of doing them one by one.",
  "Can you add an option for weekly summary emails? I don't want daily ones."],
  agent=[
  "That's a popular request — I've added your vote to the feature request tracker.",
  "Good news: that's on our roadmap for next quarter. I'll note your interest.",
  "I don't see that on the roadmap yet, but I'll submit it to the product team."],
  followups=[
  "Lots of people in the community forum are asking for the same thing.",
  "Is there a beta program I could join to try it early?"],
  closings=["You'll be notified when it launches. Thanks for the suggestion!"]),
]

def make_conversation(i):
    t = random.choice(TOPICS)
    turns = [f"Customer: {random.choice(t['openers'])}",
             f"Agent: {random.choice(t['agent'])}"]
    if random.random() < 0.7:
        turns.append(f"Customer: {random.choice(t['followups'])}")
    if random.random() < 0.5:
        turns.append(f"Agent: {random.choice(t['closings'])}")
    return {
        "conversation_id": f"conv_{i:05d}",
        "transcript": "\n".join(turns),
        "channel": random.choice(["chat", "chat", "email", "phone"]),
        "created_at": datetime.now() - timedelta(days=random.randint(0, 90),
                                                hours=random.randint(0, 23)),
    }

df = pd.DataFrame([make_conversation(i) for i in range(1, N + 1)])

client = bigquery.Client(project=PROJECT)
dest = f"{PROJECT}.{DATASET}.{DEST_TABLE}"
client.load_table_from_dataframe(
    df, dest,
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
).result()
print(f"Loaded {len(df)} synthetic conversations to {dest}\n")
print(df["transcript"].iloc[0])
