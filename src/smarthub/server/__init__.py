"""Online-serving surface for SmartHub (the FastAPI bid-recommendation API).

Kept separate from ``train_and_predict`` (which owns pulling data, building
features, training, and the model registry) so the serving process's
dependency footprint and deployment lifecycle are independent of the
training/orchestration pipeline -- see ``docker/Dockerfile.serve`` and
``docker-compose.prefect.yml``'s ``serve``/``nginx`` services.
"""
