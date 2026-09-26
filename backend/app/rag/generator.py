import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


class GeminiGenerator:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.1-flash-lite"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured"
            )

        self.client = genai.Client(
            api_key=api_key
        )

    @staticmethod
    def build_context(results):

        parts = []

        for i, result in enumerate(results, 1):

            parts.append(
                f"""SOURCE {i}
Title: {result["title"]}
Section: {result["section"]}
Document ID: {result["document_id"]}

Content:
{result["text"]}
"""
            )

        return "\n\n".join(parts)

    def generate(self, question, results):

        context = self.build_context(results)

        prompt = f"""
You are a plant-health information assistant.

Answer the user's question using only the retrieved CABI
information below.

Rules:
- Do not invent facts.
- Use the retrieved information as the factual basis.
- If the sources are insufficient, say so.
- Give a clear and practical answer.
- Do not claim to diagnose an image unless a separate
  classifier has provided a prediction.
- Do not invent pesticide doses or treatment instructions.
- Mention the relevant source titles at the end.

Retrieved CABI information:

{context}

User question:

{question}

Answer:
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return response.text