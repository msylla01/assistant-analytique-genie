import sys, csv, random
from datetime import datetime, timedelta, timezone

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
random.seed(42)

channels = ["web","mobile"]
statuses = ["approved","declined"]
countries = ["CI"]
merchants = [f"M{str(i).zfill(3)}" for i in range(1, 21)]
devices = ["android","ios","windows","mac","linux"]

now = datetime(2025,9,21,12,0,0,tzinfo=timezone.utc)
start = now - timedelta(days=7)

with open("data/transactions_sample.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["transaction_id","customer_id","merchant_id","amount","currency","timestamp","country","channel","status","ip","device"])
    for i in range(1, N+1):
        ts = start + timedelta(seconds=random.randint(0, 7*24*3600))
        amt = round(max(1, random.gauss(120, 80)), 2)
        row = [
            f"T{str(i).zfill(6)}",
            f"C{str(random.randint(1, 500)).zfill(4)}",
            random.choice(merchants),
            amt, "XOF",
            ts.isoformat().replace("+00:00","Z"),
            random.choice(countries),
            random.choice(channels),
            random.choices(statuses, weights=[0.85, 0.15])[0],
            f"102.134.22.{random.randint(1,254)}",
            random.choice(devices),
        ]
        w.writerow(row)
print(f"Fichier généré: data/transactions_sample.csv ({N} lignes)")
