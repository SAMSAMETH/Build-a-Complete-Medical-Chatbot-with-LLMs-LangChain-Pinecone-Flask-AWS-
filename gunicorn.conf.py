# gunicorn.conf.py
workers = 1
threads = 1
worker_class = "sync"
timeout = 120
keepalive = 2
preload_app = False  # Critical for memory savings