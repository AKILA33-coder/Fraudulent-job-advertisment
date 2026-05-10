"""
TrueHire AI - Dataset Preparation
Generates synthetic training data for fake job detection
(Falls back to synthetic data if Kaggle is unavailable)
"""

import pandas as pd
import numpy as np
import json, os, random
from datetime import datetime

random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ── Legitimate job templates ──────────────────────────────────────────────────
LEGIT_TITLES = [
    "Software Engineer", "Data Analyst", "Product Manager", "Frontend Developer",
    "Backend Developer", "ML Engineer", "DevOps Engineer", "QA Engineer",
    "UI/UX Designer", "Business Analyst", "Data Scientist", "Cloud Architect",
    "Mobile Developer (Android)", "Full Stack Developer", "System Administrator",
    "Database Administrator", "Cybersecurity Analyst", "Network Engineer",
    "Technical Writer", "Scrum Master",
]
LEGIT_COMPANIES = [
    "Zoho Corporation", "Infosys Limited", "TCS", "Wipro Technologies",
    "HCL Technologies", "Tech Mahindra", "Cognizant", "Accenture India",
    "IBM India", "Microsoft India", "Google India", "Amazon India",
    "Flipkart", "Swiggy", "Zomato", "Paytm", "Razorpay", "Freshworks",
    "BrowserStack", "Chargebee",
]
LEGIT_LOCATIONS = [
    "Bengaluru, Karnataka", "Mumbai, Maharashtra", "Chennai, Tamil Nadu",
    "Hyderabad, Telangana", "Pune, Maharashtra", "Delhi NCR",
    "Coimbatore, Tamil Nadu", "Kolkata, West Bengal", "Ahmedabad, Gujarat",
]

def make_legit_job(i):
    title   = random.choice(LEGIT_TITLES)
    company = random.choice(LEGIT_COMPANIES)
    loc     = random.choice(LEGIT_LOCATIONS)
    exp     = random.randint(1, 8)
    sal_min = random.choice([4, 6, 8, 10, 12, 15, 18, 20])
    sal_max = sal_min + random.choice([2, 4, 6])
    skills  = random.sample(["Python","Java","SQL","React","Node.js","AWS","Docker",
                              "Kubernetes","ML","TensorFlow","Git","REST APIs"], k=4)
    email   = f"careers@{company.split()[0].lower()}.com"
    return {
        "job_id":    i,
        "title":     title,
        "company":   company,
        "location":  loc,
        "description": (
            f"{title} — {company}\n\n"
            f"Location: {loc}\n"
            f"Experience: {exp}+ years\n"
            f"Salary: ₹{sal_min}–{sal_max} LPA\n\n"
            f"Responsibilities:\n"
            f"• Design and develop scalable software solutions\n"
            f"• Collaborate with cross-functional teams\n"
            f"• Write clean, maintainable code and documentation\n"
            f"• Participate in code reviews and agile sprints\n\n"
            f"Requirements:\n"
            f"• {exp}+ years of professional experience\n"
            f"• Strong proficiency in {', '.join(skills[:2])}\n"
            f"• Experience with {', '.join(skills[2:])}\n"
            f"• Bachelor's degree in Computer Science or related field\n\n"
            f"Apply: Send your CV to {email}\n"
            f"Website: www.{company.split()[0].lower()}.com/careers"
        ),
        "fraudulent": 0
    }

# ── Fake/scam job templates ───────────────────────────────────────────────────
SCAM_TITLES = [
    "Work From Home — Earn ₹50,000/Month!",
    "Data Entry Operator — No Experience Needed",
    "Online Part-Time Job — Guaranteed Income",
    "Home Based Packing Job — Immediate Joining",
    "Earn Money Daily — Reseller Opportunity",
    "Digital Marketing Executive — 100% Work From Home",
    "Customer Support — No Interview Required",
    "Typing Job — Earn While You Sleep",
    "Amazon/Flipkart Product Reviewer — Earn Daily",
    "Social Media Manager — No Degree Required",
]
SCAM_COMPANIES = [
    "GlobalTech Solutions", "DigiWork India", "HomeEarners Pvt Ltd",
    "QuickCash Jobs", "EasyMoney Online", "WorkFromHome.co",
    "NetProfits India", "DailyEarners Hub", "ClickJob Solutions",
    "FastHire Network",
]
SCAM_PHRASES = [
    "Pay a one-time registration fee of ₹499 to get started.",
    "No experience or qualification needed — anyone can do it!",
    "Earn ₹15,000–₹50,000 per month working just 2 hours a day!",
    "Limited slots available — apply immediately before it's too late!",
    "Contact us only on WhatsApp: +91-XXXXXXXXXX",
    "100% guaranteed income from day one!",
    "No interview required — instant selection!",
    "Work from home and earn daily cash in your account!",
    "Refer friends and earn extra ₹1000 per referral!",
    "Training fee ₹999 — refundable after 30 days of work.",
    "Contact via Telegram only for faster response.",
    "Send ₹250 security deposit to confirm your slot.",
]

def make_scam_job(i):
    title   = random.choice(SCAM_TITLES)
    company = random.choice(SCAM_COMPANIES)
    phrases = random.sample(SCAM_PHRASES, k=random.randint(3, 6))
    exclaim = "!" * random.randint(2, 6)
    return {
        "job_id":    i,
        "title":     title + exclaim,
        "company":   company,
        "location":  "Work From Home (Pan India)",
        "description": (
            f"🚨 URGENT HIRING{exclaim} — {title}\n\n"
            f"Company: {company}\n"
            f"{''.join(phrases[:2])}\n\n"
            f"{''.join(phrases[2:4])}\n\n"
            f"Job Details:\n"
            f"{''.join(phrases[4:])}\n\n"
            f"{'!!!!! ' * random.randint(1,3)}\n"
            f"APPLY NOW — Don't miss this golden opportunity{exclaim}"
        ),
        "fraudulent": 1
    }

def generate_dataset(n_legit=5000, n_fake=5000):
    print(f"Generating {n_legit} legitimate + {n_fake} fake job postings...")
    rows = []
    for i in range(n_legit):
        rows.append(make_legit_job(i))
    for i in range(n_fake):
        rows.append(make_scam_job(n_legit + i))
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    out = os.path.join(DATA_DIR, 'fake_jobs_raw.csv')
    df.to_csv(out, index=False)
    print(f"✓ Dataset saved → {out}")
    print(f"  Total: {len(df)} rows | Fake: {df['fraudulent'].sum()} | Legit: {(df['fraudulent']==0).sum()}")
    # Save summary
    summary = {
        "total":      len(df),
        "fake":       int(df['fraudulent'].sum()),
        "legitimate": int((df['fraudulent']==0).sum()),
        "generated":  datetime.now().isoformat(),
        "columns":    list(df.columns)
    }
    with open(os.path.join(DATA_DIR, 'dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved → {DATA_DIR}/dataset_summary.json")
    return df

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  TrueHire AI — Data Preparation")
    print("="*55)
    df = generate_dataset()
    print("\n✅ Data preparation complete!")
    print(f"   Next step: python ml/train.py\n")
