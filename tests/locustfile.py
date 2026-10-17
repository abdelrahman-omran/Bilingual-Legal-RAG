import random
from locust import HttpUser, between, task

QUESTIONS = [
    "What does Article 147 say about contracts?",
    "ما هي شروط صحة التعاقب؟",
    "What is the legal age of majority under Article 44?",
]


class RAGUser(HttpUser):
    # Simulates realistic user pause between requests (0.5s - 2s)
    wait_time = between(0.5, 2.0)

    @task
    def ask(self):
        # Pick a random question for every request to test search/cache variance
        question = random.choice(QUESTIONS)
        
        self.client.post(
            "/ask",
            json={"question": question},
            headers={"Content-Type": "application/json"},
            name="/ask"  # Groups metrics under a clean endpoint name in Locust UI
        )