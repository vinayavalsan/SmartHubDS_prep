import os
import pandas as pd

from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder
import redshift_connector

load_dotenv()

SSH_HOST = os.getenv("SSH_HOST")
SSH_USER = os.getenv("SSH_USER")
SSH_PRIVATE_KEY_PATH = os.path.expanduser(os.getenv("SSH_PRIVATE_KEY_PATH"))

REDSHIFT_HOST = os.getenv("REDSHIFT_HOST")
REDSHIFT_PORT = int(os.getenv("REDSHIFT_PORT", "5439"))
REDSHIFT_DB = os.getenv("REDSHIFT_DB")
REDSHIFT_USER = os.getenv("REDSHIFT_USER")
REDSHIFT_PASSWORD = os.getenv("REDSHIFT_PASSWORD")

min_created_at = "2026-06-07 00:00:00"
max_created_at = "2026-06-20 00:00:00"

# leads_query = f"""
# SELECT
#    lp.id,
#    lp.created_at,
#    lp.lead_type_id,
#    lp.state,
#    lp.zip,
#    lp.age,
#    lp.num_vehicles,
#    lp.num_drivers,
#    lp.insured,
#    lp.campaign_id,
#    lp.num_auto_violations,
#    lp.num_auto_accidents,
#    lp.continuous_coverage_months,
#    lp.home_owner,
#    lp.lead_created_at,
#    lp.dui,
#    lp.won,
#    lp.bid,
#    lp.rev,
#    lpl.lead_ping_id,
#    lpl.selected,
#    lpl.excluded,
#    lpl.est_payout,
#    lpl.payout
# FROM lead_pings lp
# LEFT JOIN lead_ping_listings lpl
#    ON lp.id = lpl.lead_ping_id
# WHERE lp.created_at >= '{min_created_at}'
#  AND lp.created_at < '{max_created_at}'
# """
leads_query = f"""
SELECT
    lp.id,
    lp.created_at,
    lp.lead_type_id,
    lp.state,
    lp.zip,
    lp.age,
    lp.num_vehicles,
    lp.num_drivers,
    lp.insured,
    lp.campaign_id,
    lp.num_auto_violations,
    lp.num_auto_accidents,
    lp.continuous_coverage_months,
    lp.home_owner,
    lp.lead_created_at,
    lp.dui,
    lp.won,
    lp.bid,
    lp.rev
FROM lead_pings lp
WHERE lp.created_at >= '{min_created_at}'
  AND lp.created_at < '{max_created_at}'
"""


with SSHTunnelForwarder(
    (SSH_HOST, 22),
    ssh_username=SSH_USER,
    ssh_pkey=SSH_PRIVATE_KEY_PATH,
    remote_bind_address=(REDSHIFT_HOST, REDSHIFT_PORT),
) as tunnel:

    print(f"Tunnel established on localhost:{tunnel.local_bind_port}")

    conn = redshift_connector.connect(
        host="localhost",
        port=tunnel.local_bind_port,
        database=REDSHIFT_DB,
        user=REDSHIFT_USER,
        password=REDSHIFT_PASSWORD,
        timeout=10,
    )

    leads_df = pd.read_sql(leads_query, conn)

    print(f"leads_df: {leads_df.shape}")
    print(sorted(leads_df.keys()))
    print(leads_df.head())

    conn.close()


leads_df.to_parquet("leads.parquet", index=False)
