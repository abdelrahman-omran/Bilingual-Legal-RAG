from locust import HttpUser, between, task

QUESTIONS = [
    "What does Article 147 say about contracts?",
    "ما هي شروط صحة التعاقب؟",
    "What is the legal age of majority under Article 44?",
]


class RAGUser(HttpUser):
    wait_time = between(0.5, 2)

    @task
    def ask(self):
        self.client.post("/ask", json={"question": self.question})

    def on_start(self):
        self.question = QUESTIONS[self.environment.runner.user_count % len(QUESTIONS)]