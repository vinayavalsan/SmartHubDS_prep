"""SmartHub bid-API replay/load test harness (tests/sim).

Run the modules as scripts (they add their own dir to sys.path):
    extract_snapshot.py  -- pull lead_pings -> snapshot.parquet   (piece 1)
    payload_builder.py   -- row -> BidRequest body                (piece 2)
    arrivals.py          -- bursty arrival schedule               (piece 3)
    sender.py            -- async POST + capture ledger           (piece 4)
    stub_server.py       -- safe fake /recommend_bid              (piece 5)
    run.py               -- orchestrator (2+3+4) -> ledger
    monitors.py          -- host/container/DB health CSV          (piece 6)
    supervisor.py        -- week-long runner (resume/heartbeat)   (piece 7)
    analyze.py           -- ledger -> latency/errors/parity       (piece 8)
See README.md.
"""
