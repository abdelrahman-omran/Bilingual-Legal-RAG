# test_langfuse.py
import os

from langfuse import Langfuse

os.environ["LANGFUSE_DEBUG"] = "True"

langfuse = Langfuse(
    public_key="pk-lf-0d63557e-f4d9-4939-9efd-6141fbafb118",
    secret_key="sk-lf-b3c3b965-6394-4096-82b7-57d4d3476445",
    host="http://localhost:3001",
)

trace = langfuse.trace(name="manual_test_trace")
trace.span(name="ping")
langfuse.flush()

print("Trace sent successfully!")